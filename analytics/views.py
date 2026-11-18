"""Reports & Analytics views."""

import json

from django.shortcuts import render
from django.views.decorators.http import require_GET

from .exports import export_history_csv, export_metrics_csv, export_pdf_report
from .services import build_analytics_context


@require_GET
def reports(request):
    preset = request.GET.get('preset', 'all')
    date_from = request.GET.get('from', '')
    date_to = request.GET.get('to', '')

    ctx = build_analytics_context(preset, date_from, date_to)

    # JSON blobs for Chart.js
    chart_payload = {
        'fake_vs_real': ctx['fake_vs_real'],
        'sentiment_dist': ctx['sentiment_dist'],
        'fake_trends': ctx['fake_trends'],
        'sentiment_trends': ctx['sentiment_trends'],
        'monthly_counts': ctx['monthly_counts'],
        'monthly_sentiment': ctx['monthly_sentiment'],
        'model_comparison': {
            'fake_news_models': [
                {
                    'name': m['name'],
                    'algorithm': m['algorithm'],
                    'accuracy': m['accuracy'],
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'f1_score': m['f1_score'],
                    'feature_method': m['feature_method'],
                    'dataset_name': m['dataset_name'],
                }
                for m in ctx['model_comparison']['fake_news_models']
            ],
            'sentiment_models': [
                {
                    'name': m['name'],
                    'algorithm': m['algorithm'],
                    'accuracy': m['accuracy'],
                    'precision': m['precision'],
                    'recall': m['recall'],
                    'f1_score': m['f1_score'],
                    'feature_method': m['feature_method'],
                    'dataset_name': m['dataset_name'],
                }
                for m in ctx['model_comparison']['sentiment_models']
            ],
        },
    }

    context = {
        'page_title': 'Reports & Analytics',
        'active_menu': 'analytics',
        'breadcrumbs': [{'name': 'Reports & Analytics', 'url': None}],
        **ctx,
        'chart_json': json.dumps(chart_payload),
        'query_params': _qs(preset, date_from, date_to),
    }
    return render(request, 'analytics/index.html', context)


@require_GET
def export_csv(request):
    return export_history_csv(
        preset=request.GET.get('preset', 'all'),
        date_from=request.GET.get('from'),
        date_to=request.GET.get('to'),
    )


@require_GET
def export_metrics(request):
    return export_metrics_csv()


@require_GET
def export_pdf(request):
    return export_pdf_report(
        preset=request.GET.get('preset', 'all'),
        date_from=request.GET.get('from'),
        date_to=request.GET.get('to'),
    )


def _qs(preset, date_from, date_to):
    parts = []
    if preset:
        parts.append(f'preset={preset}')
    if date_from:
        parts.append(f'from={date_from}')
    if date_to:
        parts.append(f'to={date_to}')
    return '&'.join(parts)
