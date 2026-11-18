"""
Dashboard statistics and chart data services.
Uses aggregation — does not load full record sets unnecessarily.
"""

from datetime import timedelta
from collections import defaultdict

from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from django.utils import timezone

from history.models import Analysis


def _apply_date_filter(qs, date_from=None, date_to=None):
    if date_from:
        qs = qs.filter(created_at__date__gte=date_from)
    if date_to:
        qs = qs.filter(created_at__date__lte=date_to)
    return qs


def get_dashboard_stats(date_from=None, date_to=None):
    """
    Compute high-level statistics.

    Counting rules:
    - total_news  = number of Analysis records (combined counts as 1)
    - real / fake = only records that have a fake_label set
    - sentiment counts = only records that have a sentiment_label set
    """
    qs = _apply_date_filter(Analysis.objects.all(), date_from, date_to)

    aggregates = qs.aggregate(
        total=Count('id'),
        real=Count('id', filter=Q(fake_label='real')),
        fake=Count('id', filter=Q(fake_label='fake')),
        positive=Count('id', filter=Q(sentiment_label='positive')),
        negative=Count('id', filter=Q(sentiment_label='negative')),
        neutral=Count('id', filter=Q(sentiment_label='neutral')),
        fake_news_type=Count('id', filter=Q(analysis_type='fake_news')),
        sentiment_type=Count('id', filter=Q(analysis_type='sentiment')),
        combined_type=Count('id', filter=Q(analysis_type='combined')),
    )

    total = aggregates['total'] or 0
    real = aggregates['real'] or 0
    fake = aggregates['fake'] or 0
    positive = aggregates['positive'] or 0
    negative = aggregates['negative'] or 0
    neutral = aggregates['neutral'] or 0

    labelled_fake = real + fake
    labelled_sentiment = positive + negative + neutral

    def pct(part, whole):
        if whole == 0:
            return 0.0
        return round((part / whole) * 100, 1)

    latest = qs.order_by('-created_at').values_list('created_at', flat=True).first()

    return {
        'total_news': total,
        'real_news': real,
        'fake_news': fake,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'real_pct': pct(real, labelled_fake),
        'fake_pct': pct(fake, labelled_fake),
        'positive_pct': pct(positive, labelled_sentiment),
        'negative_pct': pct(negative, labelled_sentiment),
        'neutral_pct': pct(neutral, labelled_sentiment),
        'type_fake_news': aggregates['fake_news_type'] or 0,
        'type_sentiment': aggregates['sentiment_type'] or 0,
        'type_combined': aggregates['combined_type'] or 0,
        'latest_at': latest,
        'has_data': total > 0,
    }


def get_fake_vs_real_chart(date_from=None, date_to=None):
    qs = _apply_date_filter(Analysis.objects.all(), date_from, date_to)
    data = qs.aggregate(
        real=Count('id', filter=Q(fake_label='real')),
        fake=Count('id', filter=Q(fake_label='fake')),
    )
    return {
        'labels': ['Real', 'Fake'],
        'values': [data['real'] or 0, data['fake'] or 0],
    }


def get_sentiment_chart(date_from=None, date_to=None):
    qs = _apply_date_filter(Analysis.objects.all(), date_from, date_to)
    data = qs.aggregate(
        positive=Count('id', filter=Q(sentiment_label='positive')),
        negative=Count('id', filter=Q(sentiment_label='negative')),
        neutral=Count('id', filter=Q(sentiment_label='neutral')),
    )
    return {
        'labels': ['Positive', 'Negative', 'Neutral'],
        'values': [
            data['positive'] or 0,
            data['negative'] or 0,
            data['neutral'] or 0,
        ],
    }


def get_daily_activity_chart(date_from=None, date_to=None, days=14):
    """
    Daily analysis counts for the last `days` (or within the provided range).
    Returns continuous date series filled with zeros where no activity.
    """
    today = timezone.localdate()

    if date_from and date_to:
        start = date_from
        end = date_to
    else:
        end = today
        start = end - timedelta(days=days - 1)

    qs = (
        Analysis.objects
        .filter(created_at__date__gte=start, created_at__date__lte=end)
        .annotate(day=TruncDate('created_at'))
        .values('day')
        .annotate(count=Count('id'))
        .order_by('day')
    )

    counts = {row['day']: row['count'] for row in qs}

    labels = []
    values = []
    current = start
    while current <= end:
        labels.append(current.strftime('%b %d'))
        values.append(counts.get(current, 0))
        current += timedelta(days=1)

    return {
        'labels': labels,
        'values': values,
    }


def get_model_usage_chart(date_from=None, date_to=None):
    """
    Count model usage across both fake_model and sentiment_model fields.
    A combined analysis that used two different models contributes to both.
    """
    qs = _apply_date_filter(Analysis.objects.all(), date_from, date_to)

    model_counts = defaultdict(int)

    for field in ('fake_model', 'sentiment_model'):
        rows = (
            qs.exclude(**{field: ''})
            .values(field)
            .annotate(c=Count('id'))
        )
        for row in rows:
            key = row[field]
            if key:
                model_counts[key] += row['c']

    # Friendly labels
    label_map = {
        'naive_bayes': 'Naïve Bayes',
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'svm': 'SVM',
        'other': 'Other',
    }

    # Stable order
    order = ['naive_bayes', 'logistic_regression', 'random_forest', 'svm', 'other']
    labels = []
    values = []
    for key in order:
        if model_counts.get(key, 0) > 0 or True:  # always show main models
            labels.append(label_map.get(key, key))
            values.append(model_counts.get(key, 0))

    # Also include any unexpected keys
    for key, cnt in model_counts.items():
        if key not in order:
            labels.append(label_map.get(key, key))
            values.append(cnt)

    return {
        'labels': labels,
        'values': values,
    }


def get_recent_analyses(limit=10, date_from=None, date_to=None):
    qs = _apply_date_filter(Analysis.objects.all(), date_from, date_to)
    return list(qs.select_related()[:limit])
