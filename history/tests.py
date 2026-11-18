"""Tests for Analysis History module."""

from datetime import timedelta

from django.test import Client, TestCase
from django.urls import reverse
from django.utils import timezone

from history.models import Analysis


class HistoryModelTests(TestCase):
    def test_create_all_types(self):
        for t in ('fake_news', 'sentiment', 'combined'):
            a = Analysis.objects.create(
                text=f'Sample text for {t} analysis long enough.',
                headline=f'Headline {t}',
                analysis_type=t,
                fake_label='real' if t != 'sentiment' else '',
                sentiment_label='positive' if t != 'fake_news' else '',
                fake_model='logistic_regression' if t != 'sentiment' else '',
                sentiment_model='logistic_regression' if t != 'fake_news' else '',
                processing_time=0.12,
            )
            self.assertIsNotNone(a.id)
            self.assertTrue(a.short_text())


class HistoryViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        now = timezone.now()
        self.r1 = Analysis.objects.create(
            text='Government announces new infrastructure spending package for rural regions.',
            headline='Infrastructure package announced',
            analysis_type='fake_news',
            fake_label='real',
            fake_confidence=0.81,
            fake_model='logistic_regression',
            fake_model_name='Sample · logistic_regression',
            fake_scores={'real': 0.81, 'fake': 0.19},
            processing_time=0.05,
            created_at=now - timedelta(days=2),
        )
        self.r2 = Analysis.objects.create(
            text='Terrible scandal rocks the administration after leaked documents surface online.',
            headline='Scandal leaks online',
            analysis_type='sentiment',
            sentiment_label='negative',
            sentiment_confidence=0.72,
            sentiment_model='logistic_regression',
            sentiment_model_name='Sample · sentiment_lr',
            sentiment_scores={'positive': 0.1, 'negative': 0.72, 'neutral': 0.18},
            processing_time=0.04,
            created_at=now - timedelta(days=1),
        )
        self.r3 = Analysis.objects.create(
            text='Scientists report promising clinical trial results for a novel therapy.',
            headline='Clinical trial success',
            analysis_type='combined',
            fake_label='real',
            fake_confidence=0.65,
            fake_model='naive_bayes',
            fake_model_name='Sample · naive_bayes',
            fake_scores={'real': 0.65, 'fake': 0.35},
            sentiment_label='positive',
            sentiment_confidence=0.58,
            sentiment_model='logistic_regression',
            sentiment_model_name='Sample · sentiment_lr',
            sentiment_scores={'positive': 0.58, 'negative': 0.2, 'neutral': 0.22},
            processing_time=0.09,
            created_at=now,
        )

    def test_list_page(self):
        resp = self.client.get(reverse('history:index'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['total_count'], 3)
        self.assertContains(resp, 'Infrastructure package')

    def test_search_headline(self):
        resp = self.client.get(reverse('history:index'), {'q': 'Infrastructure'})
        self.assertEqual(resp.context['total_count'], 1)

    def test_search_content(self):
        resp = self.client.get(reverse('history:index'), {'q': 'clinical trial'})
        self.assertEqual(resp.context['total_count'], 1)

    def test_search_id(self):
        resp = self.client.get(reverse('history:index'), {'q': str(self.r2.id)})
        self.assertEqual(resp.context['total_count'], 1)

    def test_filter_type(self):
        resp = self.client.get(reverse('history:index'), {'type': 'combined'})
        self.assertEqual(resp.context['total_count'], 1)

    def test_filter_fake_label(self):
        resp = self.client.get(reverse('history:index'), {'fake': 'real'})
        self.assertEqual(resp.context['total_count'], 2)

    def test_filter_sentiment(self):
        resp = self.client.get(reverse('history:index'), {'sentiment': 'negative'})
        self.assertEqual(resp.context['total_count'], 1)

    def test_filter_model(self):
        resp = self.client.get(reverse('history:index'), {'model': 'naive_bayes'})
        self.assertEqual(resp.context['total_count'], 1)

    def test_combined_filters(self):
        resp = self.client.get(reverse('history:index'), {
            'type': 'combined', 'fake': 'real', 'sentiment': 'positive',
        })
        self.assertEqual(resp.context['total_count'], 1)

    def test_sort_oldest(self):
        resp = self.client.get(reverse('history:index'), {'sort': 'oldest'})
        ids = [o.id for o in resp.context['page_obj']]
        self.assertEqual(ids[0], self.r1.id)

    def test_sort_latest(self):
        resp = self.client.get(reverse('history:index'), {'sort': 'latest'})
        ids = [o.id for o in resp.context['page_obj']]
        self.assertEqual(ids[0], self.r3.id)

    def test_pagination(self):
        for i in range(20):
            Analysis.objects.create(
                text=f'Extra record number {i} with enough text content.',
                headline=f'Extra {i}',
                analysis_type='fake_news',
                fake_label='fake',
            )
        resp = self.client.get(reverse('history:index'), {'page_size': 10})
        self.assertEqual(len(resp.context['page_obj']), 10)
        self.assertTrue(resp.context['page_obj'].has_next())

    def test_detail_page(self):
        resp = self.client.get(reverse('history:detail', args=[self.r3.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Clinical trial success')
        self.assertContains(resp, 'Fake News Detection')
        self.assertContains(resp, 'Sentiment Analysis')

    def test_detail_404(self):
        resp = self.client.get(reverse('history:detail', args=[99999]))
        self.assertEqual(resp.status_code, 404)

    def test_delete_individual(self):
        rid = self.r1.id
        resp = self.client.post(reverse('history:delete', args=[rid]))
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(Analysis.objects.filter(pk=rid).exists())
        self.assertEqual(Analysis.objects.count(), 2)

    def test_delete_all_requires_confirm(self):
        resp = self.client.post(reverse('history:delete_all'), {'confirm': 'wrong'})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(Analysis.objects.count(), 3)

    def test_delete_all_confirmed(self):
        resp = self.client.post(reverse('history:delete_all'), {'confirm': 'delete all'})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(Analysis.objects.count(), 0)

    def test_export_json(self):
        resp = self.client.get(reverse('history:export', args=[self.r1.id]), {'format': 'json'})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('application/json', resp['Content-Type'])

    def test_export_csv(self):
        resp = self.client.get(reverse('history:export', args=[self.r2.id]), {'format': 'csv'})
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp['Content-Type'])

    def test_export_list(self):
        resp = self.client.get(reverse('history:export_list'))
        self.assertEqual(resp.status_code, 200)
        self.assertIn('text/csv', resp['Content-Type'])

    def test_empty_history(self):
        Analysis.objects.all().delete()
        resp = self.client.get(reverse('history:index'))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['total_count'], 0)
        self.assertContains(resp, 'No analysis history')
