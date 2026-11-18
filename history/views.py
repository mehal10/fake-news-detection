"""
Analysis History — list, search, filter, sort, paginate, detail, delete, export.
"""

import csv
import json
from datetime import datetime, time
from io import StringIO

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods, require_POST

from .models import Analysis

PAGE_SIZE_OPTIONS = (10, 15, 25, 50, 100)
SORT_OPTIONS = {
    'latest': '-created_at',
    'oldest': 'created_at',
    'alpha': 'headline',
    'alpha_desc': '-headline',
}


def _parse_date(value, end_of_day=False):
    """Parse YYYY-MM-DD into aware datetime; return None on failure."""
    if not value:
        return None
    try:
        d = datetime.strptime(value.strip(), '%Y-%m-%d').date()
        t = time(23, 59, 59, 999999) if end_of_day else time(0, 0, 0)
        return timezone.make_aware(datetime.combine(d, t))
    except (ValueError, TypeError):
        return None


def _filtered_queryset(request):
    """Apply search + filters from GET params. Returns (qs, filters_dict)."""
    qs = Analysis.objects.all()

    q = (request.GET.get('q') or '').strip()
    analysis_type = (request.GET.get('type') or '').strip()
    fake_label = (request.GET.get('fake') or '').strip()
    sentiment = (request.GET.get('sentiment') or '').strip()
    model = (request.GET.get('model') or '').strip()
    date_from = (request.GET.get('from') or '').strip()
    date_to = (request.GET.get('to') or '').strip()
    sort = (request.GET.get('sort') or 'latest').strip()

    if q:
        # Search headline, content, and exact ID if numeric
        id_q = Q()
        if q.isdigit():
            id_q = Q(pk=int(q))
        qs = qs.filter(
            id_q
            | Q(headline__icontains=q)
            | Q(text__icontains=q)
        )

    if analysis_type in ('fake_news', 'sentiment', 'combined'):
        qs = qs.filter(analysis_type=analysis_type)

    if fake_label in ('real', 'fake'):
        qs = qs.filter(fake_label=fake_label)

    if sentiment in ('positive', 'negative', 'neutral'):
        qs = qs.filter(sentiment_label=sentiment)

    if model:
        qs = qs.filter(
            Q(fake_model=model)
            | Q(sentiment_model=model)
            | Q(fake_model_name__icontains=model)
            | Q(sentiment_model_name__icontains=model)
        )

    dt_from = _parse_date(date_from, end_of_day=False)
    dt_to = _parse_date(date_to, end_of_day=True)
    if dt_from:
        qs = qs.filter(created_at__gte=dt_from)
    if dt_to:
        qs = qs.filter(created_at__lte=dt_to)

    order = SORT_OPTIONS.get(sort, '-created_at')
    qs = qs.order_by(order, '-id')

    filters = {
        'q': q,
        'type': analysis_type,
        'fake': fake_label,
        'sentiment': sentiment,
        'model': model,
        'from': date_from,
        'to': date_to,
        'sort': sort,
    }
    return qs, filters


def _query_string(filters, **overrides):
    """Build a query string preserving current filters."""
    data = {**filters, **overrides}
    parts = []
    for k, v in data.items():
        if v is None or v == '':
            continue
        parts.append(f'{k}={v}')
    return '&'.join(parts)


@require_http_methods(['GET'])
def history_list(request):
    qs, filters = _filtered_queryset(request)

    try:
        page_size = int(request.GET.get('page_size', 15))
    except (TypeError, ValueError):
        page_size = 15
    if page_size not in PAGE_SIZE_OPTIONS:
        page_size = 15

    paginator = Paginator(qs, page_size)
    page_obj = paginator.get_page(request.GET.get('page'))

    # Distinct model values for filter dropdown
    model_values = sorted(
        set(
            list(
                Analysis.objects.exclude(fake_model='')
                .values_list('fake_model', flat=True)
                .distinct()
            )
            + list(
                Analysis.objects.exclude(sentiment_model='')
                .values_list('sentiment_model', flat=True)
                .distinct()
            )
        )
    )

    context = {
        'page_title': 'Analysis History',
        'active_menu': 'history',
        'breadcrumbs': [{'name': 'Analysis History', 'url': None}],
        'page_obj': page_obj,
        'filters': filters,
        'query_string': _query_string(filters),
        'total_count': paginator.count,
        'page_size': page_size,
        'page_size_options': PAGE_SIZE_OPTIONS,
        'model_values': model_values,
        'sort_options': [
            ('latest', 'Latest First'),
            ('oldest', 'Oldest First'),
            ('alpha', 'Alphabetical A–Z'),
            ('alpha_desc', 'Alphabetical Z–A'),
        ],
    }
    return render(request, 'history/index.html', context)


@require_http_methods(['GET'])
def history_detail(request, pk):
    record = get_object_or_404(Analysis, pk=pk)
    context = {
        'page_title': f'Analysis #{record.id}',
        'active_menu': 'history',
        'breadcrumbs': [
            {'name': 'Analysis History', 'url': '/history/'},
            {'name': f'#{record.id}', 'url': None},
        ],
        'record': record,
    }
    return render(request, 'history/detail.html', context)


@require_POST
def history_delete(request, pk):
    record = get_object_or_404(Analysis, pk=pk)
    rid = record.id
    record.delete()
    messages.success(request, f'Analysis #{rid} has been deleted.')
    next_url = request.POST.get('next') or '/history/'
    return redirect(next_url)


@require_POST
def history_delete_all(request):
    """Delete all history records after explicit confirmation."""
    confirm = (request.POST.get('confirm') or '').strip().lower()
    if confirm != 'delete all':
        messages.error(
            request,
            'Delete All cancelled. Type "delete all" exactly to confirm.',
        )
        return redirect('history:index')

    count, _ = Analysis.objects.all().delete()
    messages.success(request, f'Deleted {count} analysis record(s). History is now empty.')
    return redirect('history:index')


@require_http_methods(['GET'])
def history_export(request, pk):
    """Export a single analysis as JSON or CSV."""
    record = get_object_or_404(Analysis, pk=pk)
    fmt = (request.GET.get('format') or 'json').lower()
    data = record.to_export_dict()

    if fmt == 'csv':
        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow(data.keys())
        writer.writerow([
            json.dumps(v) if isinstance(v, (dict, list)) else v
            for v in data.values()
        ])
        response = HttpResponse(buf.getvalue(), content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="analysis_{pk}.csv"'
        return response

    response = HttpResponse(
        json.dumps(data, indent=2, default=str),
        content_type='application/json',
    )
    response['Content-Disposition'] = f'attachment; filename="analysis_{pk}.json"'
    return response


@require_http_methods(['GET'])
def history_export_list(request):
    """Export the current filtered list as CSV."""
    qs, filters = _filtered_queryset(request)
    qs = qs[:5000]  # safety cap

    buf = StringIO()
    writer = csv.writer(buf)
    headers = [
        'id', 'analysis_type', 'headline', 'fake_label', 'fake_confidence',
        'sentiment_label', 'sentiment_confidence', 'fake_model', 'sentiment_model',
        'processing_time', 'created_at',
    ]
    writer.writerow(headers)
    for r in qs:
        writer.writerow([
            r.id, r.analysis_type, r.headline, r.fake_label, r.fake_confidence,
            r.sentiment_label, r.sentiment_confidence, r.fake_model, r.sentiment_model,
            r.processing_time, r.created_at.isoformat() if r.created_at else '',
        ])

    response = HttpResponse(buf.getvalue(), content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="analysis_history.csv"'
    return response
