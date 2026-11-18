from django.test import Client, TestCase
from django.urls import reverse

from history.models import Analysis


class DashboardTests(TestCase):
    def setUp(self):
        self.client = Client()
        Analysis.objects.create(
            text='Dashboard sample article text long enough for storage.',
            headline='Dash sample',
            analysis_type='fake_news',
            fake_label='real',
        )

    def test_dashboard_loads(self):
        resp = self.client.get(reverse('dashboard:index'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Total')

    def test_stats_reflect_data(self):
        from dashboard.services import get_dashboard_stats
        stats = get_dashboard_stats()
        self.assertGreaterEqual(stats['total_news'], 1)
        self.assertGreaterEqual(stats['real_news'], 1)
