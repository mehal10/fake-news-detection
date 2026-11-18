"""Tests for Reports & Analytics module."""

from datetime import timedelta

from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from history.models import Analysis
from analytics.services import (
    build_analytics_context,
    get_overview_stats,
    resolve_date_range,
)


class DateRangeTests(TestCase):
    def test_presets(self):
        today = timezone.localdate()
        df, dt, label = resolve_date_range('today')
        self.assertEqual(df, today)
        self.assertEqual(dt, today)

        df, dt, _ = resolve_date_range('7d')
        self.assertEqual((dt - df).days, 6)

        df, dt, _ = resolve_date_range('all')
        self.assertIsNone(df)
        self.assertIsNone(dt)

    def test_custom_swaps_inverted(self):
        df, dt, _ = resolve_date_range('custom', '2026-11-10', '2026-11-01')
        self.assertEqual(df.isoformat(), '2026-11-01')
        self.assertEqual(dt.isoformat(), '2026-11-10')

    def test_invalid_custom(self):
        df, dt, label = resolve_date_range('custom', 'not-a-date', '')
        self.assertIsNone(df)
        self.assertEqual(label, 'All Time')


class AnalyticsStatsTests(TestCase):
    def setUp(self):
        now = timezone.now()
        Analysis.objects.create(
            text='Real positive article content here for testing.',
            headline='Good news',
            analysis_type='combined',
            fake_label='real',
            sentiment_label='positive',
            created_at=now,
        )
        Analysis.objects.create(
            text='Fake negative article content here for testing.',
            headline='Bad rumour',
            analysis_type='fake_news',
            fake_label='fake',
            created_at=now - timedelta(days=3),
        )
        Analysis.objects.create(
            text='Neutral sentiment only content for testing purposes.',
            headline='Plain update',
            analysis_type='sentiment',
            sentiment_label='neutral',
            created_at=now - timedelta(days=10),
        )

    def test_overview_all(self):
        s = get_overview_stats()
        self.assertEqual(s['total'], 3)
        self.assertEqual(s['real'], 1)
        self.assertEqual(s['fake'], 1)
        self.assertEqual(s['positive'], 1)
        self.assertEqual(s['neutral'], 1)
        self.assertTrue(s['has_data'])

    def test_overview_empty(self):
        Analysis.objects.all().delete()
        s = get_overview_stats()
        self.assertEqual(s['total'], 0)
        self.assertFalse(s['has_data'])

    def test_date_filter(self):
        today = timezone.localdate()
        s = get_overview_stats(date_from=today, date_to=today)
        self.assertEqual(s['total'], 1)

    def test_build_context(self):
        ctx = build_analytics_context('all')
        self.assertIn('stats', ctx)
        self.assertIn('fake_vs_real', ctx)
        self.assertIn('model_comparison', ctx)
        self.assertEqual(ctx['fake_vs_real']['values'][0], 1)


class AnalyticsViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        Analysis.objects.create(
            text='Sample analysis text for view testing long enough.',
            headline='View test',
            analysis_type='fake_news',
            fake_label='real',
        )

    def test_reports_page(self):
        resp = self.client.get(reverse('analytics:index'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Reports & Analytics')
        self.assertContains(resp, 'Total News Analysed')

    def test_preset_filter(self):
        resp = self.client.get(reverse('analytics:index'), {'preset': '7d'})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['preset'], '7d')

    def test_custom_range(self):
        resp = self.client.get(reverse('analytics:index'), {
            'preset': 'custom',
            'from': '2020-01-01',
            'to': '2020-01-02',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['stats']['total'], 0)

    def test_export_csv(self):
        resp = self.client.get(reverse('analytics:export_csv'))
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp['Content-Type'])
        self.assertIn(b'analysis_type', resp.content)

    def test_export_metrics_csv(self):
        resp = self.client.get(reverse('analytics:export_metrics'))
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp['Content-Type'])

    def test_export_pdf(self):
        resp = self.client.get(reverse('analytics:export_pdf'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp['Content-Type'], 'application/pdf')
        self.assertTrue(resp.content.startswith(b'%PDF'))
