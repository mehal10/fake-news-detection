import json
from datetime import datetime

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from .services import (
    get_dashboard_stats,
    get_fake_vs_real_chart,
    get_sentiment_chart,
    get_daily_activity_chart,
    get_model_usage_chart,
    get_recent_analyses,
)


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(value, '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None


def dashboard(request):
    date_from = _parse_date(request.GET.get('from'))
    date_to = _parse_date(request.GET.get('to'))

    stats = get_dashboard_stats(date_from, date_to)
    recent = get_recent_analyses(limit=10, date_from=date_from, date_to=date_to)

    chart_fake_real = get_fake_vs_real_chart(date_from, date_to)
    chart_sentiment = get_sentiment_chart(date_from, date_to)
    chart_daily = get_daily_activity_chart(date_from, date_to)
    chart_models = get_model_usage_chart(date_from, date_to)

    context = {
        'page_title': 'Dashboard',
        'active_menu': 'dashboard',
        'breadcrumbs': [
            {'name': 'Dashboard', 'url': None},
        ],
        'stats': stats,
        'recent': recent,
        'chart_fake_real_json': json.dumps(chart_fake_real),
        'chart_sentiment_json': json.dumps(chart_sentiment),
        'chart_daily_json': json.dumps(chart_daily),
        'chart_models_json': json.dumps(chart_models),
        'filter_from': request.GET.get('from', ''),
        'filter_to': request.GET.get('to', ''),
    }
    return render(request, 'dashboard/index.html', context)


@require_GET
def dashboard_api(request):
    """JSON endpoint for refresh / AJAX chart updates."""
    date_from = _parse_date(request.GET.get('from'))
    date_to = _parse_date(request.GET.get('to'))

    stats = get_dashboard_stats(date_from, date_to)
    # Make datetime JSON-serializable
    if stats.get('latest_at'):
        stats['latest_at'] = stats['latest_at'].isoformat()

    return JsonResponse({
        'stats': stats,
        'charts': {
            'fake_real': get_fake_vs_real_chart(date_from, date_to),
            'sentiment': get_sentiment_chart(date_from, date_to),
            'daily': get_daily_activity_chart(date_from, date_to),
            'models': get_model_usage_chart(date_from, date_to),
        },
    })
