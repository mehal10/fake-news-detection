from unittest.mock import MagicMock, patch

from django.test import Client, TestCase
from django.urls import reverse

from history.models import Analysis


class DetectorViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('detector:index')

    def test_get(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)

    def test_empty_post(self):
        resp = self.client.post(self.url, {'action': 'detect', 'content': '', 'headline': ''})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Analysis.objects.count(), 0)

    @patch('detector.views.get_fake_predictor')
    def test_successful_prediction_saves(self, mock_get):
        result = MagicMock()
        result.label = 'real'
        result.confidence = 0.8
        result.algorithm = 'logistic_regression'
        result.probabilities = {'real': 0.8, 'fake': 0.2}
        result.model_name = 'Test'
        result.model_id = 'm1'
        result.processing_time_sec = 0.01
        result.to_dict.return_value = {
            'label': 'real', 'confidence': 0.8, 'probabilities': {'real': 0.8, 'fake': 0.2},
            'model_name': 'Test', 'algorithm': 'logistic_regression',
            'processing_time_sec': 0.01, 'model_id': 'm1',
        }
        mock_get.return_value.predict.return_value = result

        resp = self.client.post(self.url, {
            'action': 'detect',
            'headline': 'Headline',
            'content': 'This is a long enough news article body for testing prediction save path.',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Analysis.objects.filter(analysis_type='fake_news').count(), 1)

    @patch('detector.views.get_fake_predictor')
    def test_model_missing_message(self, mock_get):
        from ml_engine.services.fake_news_predictor import ModelLoadError
        mock_get.return_value.predict.side_effect = ModelLoadError('No trained fake-news model')
        resp = self.client.post(self.url, {
            'action': 'detect',
            'content': 'This is a long enough news article body for testing error handling path.',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Analysis.objects.count(), 0)
