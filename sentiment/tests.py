from unittest.mock import MagicMock, patch

from django.test import Client, TestCase
from django.urls import reverse

from history.models import Analysis


class SentimentViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('sentiment:index')

    def test_get(self):
        self.assertEqual(self.client.get(self.url).status_code, 200)

    def test_empty_post(self):
        resp = self.client.post(self.url, {'action': 'analyse', 'content': ''})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Analysis.objects.count(), 0)

    @patch('sentiment.views.get_sentiment_predictor')
    def test_saves_on_success(self, mock_get):
        result = MagicMock()
        result.label = 'positive'
        result.confidence = 0.7
        result.algorithm = 'logistic_regression'
        result.model_name = 'Sent'
        result.model_id = 's1'
        result.probabilities = {'positive': 0.7, 'negative': 0.1, 'neutral': 0.2}
        result.processing_time_sec = 0.02
        result.to_dict.return_value = {
            'label': 'positive', 'confidence': 0.7,
            'probabilities': {'positive': 0.7, 'negative': 0.1, 'neutral': 0.2},
            'model_name': 'Sent', 'algorithm': 'logistic_regression',
            'processing_time_sec': 0.02, 'model_id': 's1', 'keywords': {},
        }
        mock_get.return_value.predict.return_value = result
        resp = self.client.post(self.url, {
            'action': 'analyse',
            'content': 'Excellent breakthrough with outstanding results for patients worldwide.',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Analysis.objects.filter(analysis_type='sentiment').count(), 1)
