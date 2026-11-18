"""
CSV and PDF export generators for Reports & Analytics.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from django.http import HttpResponse
from django.utils import timezone

from history.models import Analysis

from .services import (
    build_analytics_context,
    get_model_comparisons,
    get_overview_stats,
    resolve_date_range,
)


def export_history_csv(
    preset: str = 'all',
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> HttpResponse:
    df, dt, _ = resolve_date_range(preset, date_from, date_to)
    qs = Analysis.objects.all().order_by('-created_at')
    if df:
        qs = qs.filter(created_at__date__gte=df)
    if dt:
        qs = qs.filter(created_at__date__lte=dt)
    qs = qs[:10000]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'id', 'analysis_type', 'headline', 'fake_label', 'fake_confidence',
        'sentiment_label', 'sentiment_confidence', 'fake_model', 'sentiment_model',
        'processing_time', 'created_at',
    ])
    for r in qs:
        writer.writerow([
            r.id, r.analysis_type, r.headline, r.fake_label, r.fake_confidence,
            r.sentiment_label, r.sentiment_confidence, r.fake_model, r.sentiment_model,
            r.processing_time,
            r.created_at.isoformat() if r.created_at else '',
        ])

    resp = HttpResponse(buf.getvalue(), content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename="analysis_history_export.csv"'
    return resp


def export_metrics_csv() -> HttpResponse:
    """Export model evaluation metrics from the registry."""
    data = get_model_comparisons()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'task', 'model_id', 'name', 'algorithm', 'feature_method', 'dataset_name',
        'accuracy', 'precision', 'recall', 'f1_score',
        'train_size', 'test_size', 'training_time_sec', 'testing_time_sec',
        'is_active', 'created_at',
    ])
    for task, key in (('fake_news', 'fake_news_models'), ('sentiment', 'sentiment_models')):
        for m in data.get(key, []):
            writer.writerow([
                task, m.get('model_id'), m.get('name'), m.get('algorithm'),
                m.get('feature_method'), m.get('dataset_name'),
                m.get('accuracy'), m.get('precision'), m.get('recall'), m.get('f1_score'),
                m.get('train_size'), m.get('test_size'),
                m.get('training_time_sec'), m.get('testing_time_sec'),
                m.get('is_active'), m.get('created_at'),
            ])
    resp = HttpResponse(buf.getvalue(), content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename="model_evaluation_metrics.csv"'
    return resp


def export_pdf_report(
    preset: str = 'all',
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> HttpResponse:
    """Generate a professional PDF summary report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    ctx = build_analytics_context(preset, date_from, date_to)
    stats = ctx['stats']
    models = ctx['model_comparison']
    generated = timezone.now().strftime('%Y-%m-%d %H:%M UTC')

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title='News Analysis Report',
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle', fontSize=16, leading=20, alignment=TA_CENTER,
        spaceAfter=6, textColor=colors.HexColor('#171717'), fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='Section', fontSize=12, leading=16, spaceBefore=14, spaceAfter=6,
        textColor=colors.HexColor('#262626'), fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='BodyText2', fontSize=9, leading=12, textColor=colors.HexColor('#404040'),
    ))
    styles.add(ParagraphStyle(
        name='Muted', fontSize=8, leading=10, textColor=colors.HexColor('#737373'),
        alignment=TA_CENTER, spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name='Note', fontSize=8, leading=10, textColor=colors.HexColor('#525252'),
        spaceBefore=4, spaceAfter=8,
    ))

    story: List[Any] = []
    story.append(Paragraph('Fake News Detection & News Sentiment Analysis', styles['ReportTitle']))
    story.append(Paragraph('Reports & Analytics Summary', styles['ReportTitle']))
    story.append(Paragraph(
        f'Generated: {generated} · Period: {ctx["range_label"]}',
        styles['Muted'],
    ))

    # Overview table
    story.append(Paragraph('1. Analysis Overview', styles['Section']))
    overview_data = [
        ['Metric', 'Count', 'Share'],
        ['Total News Analysed', str(stats['total']), '—'],
        ['Real News', str(stats['real']), f"{stats['real_pct']}%"],
        ['Fake News', str(stats['fake']), f"{stats['fake_pct']}%"],
        ['Positive Sentiment', str(stats['positive']), f"{stats['positive_pct']}%"],
        ['Negative Sentiment', str(stats['negative']), f"{stats['negative_pct']}%"],
        ['Neutral Sentiment', str(stats['neutral']), f"{stats['neutral_pct']}%"],
        ['Fake News Analyses', str(stats['type_fake_news']), '—'],
        ['Sentiment Analyses', str(stats['type_sentiment']), '—'],
        ['Combined Analyses', str(stats['type_combined']), '—'],
    ]
    t = Table(overview_data, colWidths=[90 * mm, 40 * mm, 40 * mm])
    t.setStyle(_table_style())
    story.append(t)

    if not stats['has_data']:
        story.append(Paragraph(
            'No analysis records in the selected period. Charts and counts are zero.',
            styles['Note'],
        ))

    # Fake / Real distribution text
    story.append(Paragraph('2. Fake vs Real Distribution', styles['Section']))
    story.append(Paragraph(
        f"Real: {stats['real']} ({stats['real_pct']}%) · "
        f"Fake: {stats['fake']} ({stats['fake_pct']}%)",
        styles['BodyText2'],
    ))

    story.append(Paragraph('3. Sentiment Distribution', styles['Section']))
    story.append(Paragraph(
        f"Positive: {stats['positive']} ({stats['positive_pct']}%) · "
        f"Negative: {stats['negative']} ({stats['negative_pct']}%) · "
        f"Neutral: {stats['neutral']} ({stats['neutral_pct']}%)",
        styles['BodyText2'],
    ))

    # Model comparison
    story.append(Paragraph('4. Model Evaluation Comparison (Historical)', styles['Section']))
    story.append(Paragraph(models.get('note', ''), styles['Note']))

    for task_label, key in (
        ('Fake News Models', 'fake_news_models'),
        ('Sentiment Models', 'sentiment_models'),
    ):
        rows = models.get(key) or []
        if not rows:
            story.append(Paragraph(f'{task_label}: none trained.', styles['BodyText2']))
            continue
        story.append(Paragraph(task_label, styles['BodyText2']))
        header = ['Name', 'Algo', 'Features', 'Dataset', 'Acc', 'F1', 'Train(s)', 'Test(s)']
        table_data = [header]
        for m in rows:
            table_data.append([
                _trunc(m.get('name'), 22),
                _trunc(m.get('algorithm'), 12),
                m.get('feature_method') or '—',
                _trunc(m.get('dataset_name'), 16),
                _fmt(m.get('accuracy')),
                _fmt(m.get('f1_score')),
                _fmt(m.get('training_time_sec')),
                _fmt(m.get('testing_time_sec')),
            ])
        mt = Table(table_data, colWidths=[38 * mm, 24 * mm, 18 * mm, 30 * mm, 14 * mm, 14 * mm, 16 * mm, 16 * mm])
        mt.setStyle(_table_style(header=True, small=True))
        story.append(mt)
        story.append(Spacer(1, 6))

    if models.get('best_fake'):
        b = models['best_fake']
        story.append(Paragraph(
            f"Best fake-news model by F1 (within dataset): {b.get('name')} "
            f"(F1={_fmt(b.get('f1_score'))}, dataset={b.get('dataset_name')})",
            styles['BodyText2'],
        ))
    if models.get('best_sentiment'):
        b = models['best_sentiment']
        story.append(Paragraph(
            f"Best sentiment model by F1 (within dataset): {b.get('name')} "
            f"(F1={_fmt(b.get('f1_score'))}, dataset={b.get('dataset_name')})",
            styles['BodyText2'],
        ))

    # Feature method comparison
    feat = models.get('feature_compare') or []
    if feat:
        story.append(Paragraph('5. Feature Extraction Comparison', styles['Section']))
        story.append(Paragraph(
            'Averages grouped by dataset and feature method. '
            'Do not treat different datasets as equivalent.',
            styles['Note'],
        ))
        fdata = [['Dataset', 'Method', 'Models', 'Avg Acc', 'Avg F1']]
        for row in feat:
            fdata.append([
                _trunc(row['dataset_name'], 24),
                row['feature_method'],
                str(row['model_count']),
                _fmt(row['avg_accuracy']),
                _fmt(row['avg_f1']),
            ])
        ft = Table(fdata, colWidths=[50 * mm, 30 * mm, 20 * mm, 25 * mm, 25 * mm])
        ft.setStyle(_table_style())
        story.append(ft)

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        'Machine learning predictions are estimates based on training data patterns. '
        'They do not independently verify factual accuracy of news content.',
        styles['Note'],
    ))

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()

    resp = HttpResponse(pdf, content_type='application/pdf')
    resp['Content-Disposition'] = 'attachment; filename="news_analysis_report.pdf"'
    return resp


def _table_style(header: bool = True, small: bool = False) -> 'TableStyle':
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle

    font_size = 7 if small else 8
    style_cmds = [
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), font_size),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#262626')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#e5e5e5')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    return TableStyle(style_cmds)


def _fmt(v) -> str:
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)


def _trunc(v, n: int) -> str:
    s = str(v or '—')
    return s if len(s) <= n else s[: n - 1] + '…'
