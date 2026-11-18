from django.test import Client, TestCase
from django.urls import reverse


class CorePagesTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_landing(self):
        resp = self.client.get(reverse('core:landing'))
        self.assertEqual(resp.status_code, 200)

    def test_about(self):
        resp = self.client.get(reverse('core:about'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Project Overview')
        self.assertContains(resp, 'Limitations')
