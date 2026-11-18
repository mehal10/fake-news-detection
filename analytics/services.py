"""
Reports & Analytics services.
All statistics come from Analysis records and the model registry.
No artificial metrics are generated.
"""

from __future__ import annotations

from calendar import month_abbr
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from django.db.models import Count, Q
from django.db.models.functions import TruncDate, TruncMonth
from django.utils import timezone

from history.models import Analysis
from ml_engine.services.model_registry import ModelRegistry


# ── Date range helpers ──────────────────────────────────────────────────────

PRESETS = {
    'today': 'Today',
    '7d': 'Last 7 Days',
    '30d': 'Last 30 Days',
    'month': 'This Month',
    'all': 'All Time',
    'custom': 'Custom Range',
}


def resolve_date_range(
    preset: str = 'all',
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Tuple[Optional[date], Optional[date], str]:
    """
    Return (from_date, to_date, label) based on preset or custom strings.
    Invalid custom dates fall back to all-time.
    """
    today = timezone.localdate()
    preset = (preset or 'all').strip().lower()

    if preset == 'today':
        return today, today, PRESETS['today']
    if preset == '7d':
        return today - timedelta(days=6), today, PRESETS['7d']
    if preset == '30d':
        return today - timedelta(days=29), today, PRESETS['30d']
    if preset == 'month':
        start = today.replace(day=1)
        return start, today, PRESETS['month']
    if preset == 'custom':
        df = _parse_date(date_from)
        dt = _parse_date(date_to)
        if df and dt and df > dt:
            df, dt = dt, df
        if df or dt:
            label = f"{df or '…'} → {dt or '…'}"
            return df, dt, label
        return None, None, PRESETS['all']
    return None, None, PRESETS['all']


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value.strip(), '%Y-%m-%d').date()
    except (ValueError, TypeError):
        return None


def _filter_qs(date_from: Optional[date] = None, date_to: Optional[date] = None):
    qs = Analysis.objects.all()
    if date_from:
        qs = qs.filter(created_at__date__gte=date_from)
    if date_to:
        qs = qs.filter(created_at__date__lte=date_to)
    return qs


def _pct(part: int, whole: int) -> float:
    if not whole:
        return 0.0
    return round(part / whole * 100, 1)


# ── Overview statistics ─────────────────────────────────────────────────────

def get_overview_stats(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
) -> Dict[str, Any]:
    qs = _filter_qs(date_from, date_to)
    agg = qs.aggregate(
        total=Count('id'),
        real=Count('id', filter=Q(fake_label='real')),
        fake=Count('id', filter=Q(fake_label='fake')),
        positive=Count('id', filter=Q(sentiment_label='positive')),
        negative=Count('id', filter=Q(sentiment_label='negative')),
        neutral=Count('id', filter=Q(sentiment_label='neutral')),
        type_fake=Count('id', filter=Q(analysis_type='fake_news')),
        type_sent=Count('id', filter=Q(analysis_type='sentiment')),
        type_comb=Count('id', filter=Q(analysis_type='combined')),
    )
    total = agg['total'] or 0
    real = agg['real'] or 0
    fake = agg['fake'] or 0
    pos = agg['positive'] or 0
    neg = agg['negative'] or 0
    neu = agg['neutral'] or 0
    labelled_fake = real + fake
    labelled_sent = pos + neg + neu
    return {
        'total': total,
        'real': real,
        'fake': fake,
        'positive': pos,
        'negative': neg,
        'neutral': neu,
        'real_pct': _pct(real, labelled_fake),
        'fake_pct': _pct(fake, labelled_fake),
        'positive_pct': _pct(pos, labelled_sent),
        'negative_pct': _pct(neg, labelled_sent),
        'neutral_pct': _pct(neu, labelled_sent),
        'type_fake_news': agg['type_fake'] or 0,
        'type_sentiment': agg['type_sent'] or 0,
        'type_combined': agg['type_comb'] or 0,
        'has_data': total > 0,
    }


# ── Fake news analytics ─────────────────────────────────────────────────────

def get_fake_vs_real(date_from=None, date_to=None) -> Dict[str, Any]:
    s = get_overview_stats(date_from, date_to)
    return {
        'labels': ['Real', 'Fake'],
        'values': [s['real'], s['fake']],
    }


def get_daily_fake_real_trends(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    days: int = 30,
) -> Dict[str, Any]:
    today = timezone.localdate()
    end = date_to or today
    start = date_from or (end - timedelta(days=days - 1))

    qs = (
        Analysis.objects.filter(created_at__date__gte=start, created_at__date__lte=end)
        .exclude(fake_label='')
        .annotate(day=TruncDate('created_at'))
        .values('day', 'fake_label')
        .annotate(c=Count('id'))
        .order_by('day')
    )
    real_map: Dict[date, int] = defaultdict(int)
    fake_map: Dict[date, int] = defaultdict(int)
    for row in qs:
        if row['fake_label'] == 'real':
            real_map[row['day']] += row['c']
        elif row['fake_label'] == 'fake':
            fake_map[row['day']] += row['c']

    labels, real_vals, fake_vals = [], [], []
    cur = start
    while cur <= end:
        labels.append(cur.strftime('%b %d'))
        real_vals.append(real_map.get(cur, 0))
        fake_vals.append(fake_map.get(cur, 0))
        cur += timedelta(days=1)
    return {
        'labels': labels,
        'real': real_vals,
        'fake': fake_vals,
    }


def get_monthly_counts(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    months: int = 12,
) -> Dict[str, Any]:
    today = timezone.localdate()
    end = date_to or today
    # approximate start
    if date_from:
        start = date_from.replace(day=1)
    else:
        start = (end.replace(day=1) - timedelta(days=30 * (months - 1))).replace(day=1)

    qs = (
        Analysis.objects.filter(created_at__date__gte=start, created_at__date__lte=end)
        .annotate(month=TruncMonth('created_at'))
        .values('month')
        .annotate(c=Count('id'))
        .order_by('month')
    )
    counts = {row['month'].date() if hasattr(row['month'], 'date') else row['month']: row['c'] for row in qs}

    labels, values = [], []
    # walk month by month
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (y, m) <= (end_y, end_m):
        key = date(y, m, 1)
        labels.append(f'{month_abbr[m]} {y}')
        values.append(counts.get(key, 0))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return {'labels': labels, 'values': values}


# ── Sentiment analytics ─────────────────────────────────────────────────────

def get_sentiment_distribution(date_from=None, date_to=None) -> Dict[str, Any]:
    s = get_overview_stats(date_from, date_to)
    return {
        'labels': ['Positive', 'Negative', 'Neutral'],
        'values': [s['positive'], s['negative'], s['neutral']],
    }


def get_daily_sentiment_trends(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    days: int = 30,
) -> Dict[str, Any]:
    today = timezone.localdate()
    end = date_to or today
    start = date_from or (end - timedelta(days=days - 1))

    qs = (
        Analysis.objects.filter(created_at__date__gte=start, created_at__date__lte=end)
        .exclude(sentiment_label='')
        .annotate(day=TruncDate('created_at'))
        .values('day', 'sentiment_label')
        .annotate(c=Count('id'))
        .order_by('day')
    )
    maps = {
        'positive': defaultdict(int),
        'negative': defaultdict(int),
        'neutral': defaultdict(int),
    }
    for row in qs:
        label = row['sentiment_label']
        if label in maps:
            maps[label][row['day']] += row['c']

    labels, pos, neg, neu = [], [], [], []
    cur = start
    while cur <= end:
        labels.append(cur.strftime('%b %d'))
        pos.append(maps['positive'].get(cur, 0))
        neg.append(maps['negative'].get(cur, 0))
        neu.append(maps['neutral'].get(cur, 0))
        cur += timedelta(days=1)
    return {
        'labels': labels,
        'positive': pos,
        'negative': neg,
        'neutral': neu,
    }


def get_monthly_sentiment(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    months: int = 12,
) -> Dict[str, Any]:
    today = timezone.localdate()
    end = date_to or today
    if date_from:
        start = date_from.replace(day=1)
    else:
        start = (end.replace(day=1) - timedelta(days=30 * (months - 1))).replace(day=1)

    qs = (
        Analysis.objects.filter(created_at__date__gte=start, created_at__date__lte=end)
        .exclude(sentiment_label='')
        .annotate(month=TruncMonth('created_at'))
        .values('month', 'sentiment_label')
        .annotate(c=Count('id'))
        .order_by('month')
    )
    maps = {
        'positive': defaultdict(int),
        'negative': defaultdict(int),
        'neutral': defaultdict(int),
    }
    for row in qs:
        month_key = row['month'].date() if hasattr(row['month'], 'date') else row['month']
        label = row['sentiment_label']
        if label in maps:
            maps[label][month_key] += row['c']

    labels, pos, neg, neu = [], [], [], []
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (y, m) <= (end_y, end_m):
        key = date(y, m, 1)
        labels.append(f'{month_abbr[m]} {y}')
        pos.append(maps['positive'].get(key, 0))
        neg.append(maps['negative'].get(key, 0))
        neu.append(maps['neutral'].get(key, 0))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return {
        'labels': labels,
        'positive': pos,
        'negative': neg,
        'neutral': neu,
    }


# ── Model performance (from registry — historical evaluations) ──────────────

def get_model_comparisons() -> Dict[str, Any]:
    """
    Return trained model evaluation results from the registry.
    Metrics are historical (from training time), not live prediction stats.
    Models are grouped by task; only compare models that share the same
    dataset_name when ranking "best".
    """
    registry = ModelRegistry()
    fake_models = [m for m in registry.list_models(task='fake_news') if m.status == 'ready']
    sent_models = [m for m in registry.list_models(task='sentiment') if m.status == 'ready']

    def _row(m) -> Dict[str, Any]:
        metrics = m.metrics or {}
        return {
            'model_id': m.model_id,
            'name': m.name,
            'algorithm': m.algorithm,
            'feature_method': m.feature_method,
            'dataset_name': m.dataset_name or '—',
            'dataset_id': m.dataset_id,
            'is_active': m.is_active,
            'train_size': m.train_size,
            'test_size': m.test_size,
            'training_time_sec': m.training_time_sec,
            'testing_time_sec': m.testing_time_sec,
            'created_at': m.created_at,
            'accuracy': metrics.get('accuracy'),
            'precision': metrics.get('precision'),
            'recall': metrics.get('recall'),
            'f1_score': metrics.get('f1_score'),
            'macro_f1': (metrics.get('macro_avg') or {}).get('f1-score')
            if isinstance(metrics.get('macro_avg'), dict)
            else metrics.get('macro_f1'),
            'confusion_matrix': metrics.get('confusion_matrix'),
            'labels': metrics.get('labels'),
            'cm_rows': _labelled_cm(
                metrics.get('confusion_matrix'),
                metrics.get('labels'),
            ),
            'classification_report': metrics.get('classification_report'),
            'per_class': metrics.get('per_class'),
        }

    fake_rows = [_row(m) for m in fake_models]
    sent_rows = [_row(m) for m in sent_models]

    # Best by F1 within same dataset (fake news)
    best_fake = _best_by_metric(fake_rows, 'f1_score')
    best_sent = _best_by_metric(sent_rows, 'f1_score')

    # Feature method comparison within same dataset
    feature_compare = _feature_method_compare(fake_rows)

    return {
        'fake_news_models': fake_rows,
        'sentiment_models': sent_rows,
        'best_fake': best_fake,
        'best_sentiment': best_sent,
        'feature_compare': feature_compare,
        'note': (
            'Metrics below are historical evaluation results recorded at training time. '
            'They are not derived from live prediction traffic. '
            'Only models trained on the same labelled dataset should be compared as equivalent.'
        ),
    }


def _labelled_cm(matrix, labels) -> List[Dict[str, Any]]:
    """Attach true-label names to confusion matrix rows for templates."""
    if not matrix:
        return []
    labels = list(labels or [])
    rows = []
    for i, row in enumerate(matrix):
        true_label = labels[i] if i < len(labels) else f'class_{i}'
        rows.append({'true_label': true_label, 'cells': list(row)})
    return rows


def _best_by_metric(rows: List[Dict], metric: str) -> Optional[Dict[str, Any]]:
    """Pick best model per dataset group, then overall best among those."""
    if not rows:
        return None
    by_dataset: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_dataset[r.get('dataset_name') or 'unknown'].append(r)

    winners = []
    for ds, group in by_dataset.items():
        scored = [r for r in group if r.get(metric) is not None]
        if not scored:
            continue
        best = max(scored, key=lambda r: r[metric])
        winners.append(best)
    if not winners:
        return None
    return max(winners, key=lambda r: r.get(metric) or 0)


def _feature_method_compare(rows: List[Dict]) -> List[Dict[str, Any]]:
    """
    Group by (dataset_name, feature_method) and average metrics.
    Makes TF-IDF vs Word2Vec comparison explicit without mixing datasets.
    """
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        key = (r.get('dataset_name') or 'unknown', r.get('feature_method') or 'unknown')
        groups[key].append(r)

    result = []
    for (ds, feat), group in sorted(groups.items()):
        def avg(key):
            vals = [g[key] for g in group if g.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        result.append({
            'dataset_name': ds,
            'feature_method': feat,
            'model_count': len(group),
            'avg_accuracy': avg('accuracy'),
            'avg_f1': avg('f1_score'),
            'avg_precision': avg('precision'),
            'avg_recall': avg('recall'),
            'models': [g['name'] for g in group],
        })
    return result


# ── Chart data package for the page ─────────────────────────────────────────

def build_analytics_context(
    preset: str = 'all',
    date_from_str: Optional[str] = None,
    date_to_str: Optional[str] = None,
) -> Dict[str, Any]:
    date_from, date_to, range_label = resolve_date_range(preset, date_from_str, date_to_str)
    stats = get_overview_stats(date_from, date_to)
    return {
        'preset': preset if preset in PRESETS else 'all',
        'presets': PRESETS,
        'date_from': date_from.isoformat() if date_from else '',
        'date_to': date_to.isoformat() if date_to else '',
        'range_label': range_label,
        'stats': stats,
        'fake_vs_real': get_fake_vs_real(date_from, date_to),
        'sentiment_dist': get_sentiment_distribution(date_from, date_to),
        'fake_trends': get_daily_fake_real_trends(date_from, date_to),
        'sentiment_trends': get_daily_sentiment_trends(date_from, date_to),
        'monthly_counts': get_monthly_counts(date_from, date_to),
        'monthly_sentiment': get_monthly_sentiment(date_from, date_to),
        'model_comparison': get_model_comparisons(),
    }
