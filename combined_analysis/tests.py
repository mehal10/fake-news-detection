"""
Tests for Combined News Analysis module.
Uses real predictors when models exist; mocks for controlled combination coverage.
"""

from unittest.mock import MagicMock, patch

from django.test import Client, TestCase
from django.urls import reverse

from history.models import Analysis
from ml_engine.services.combined_predictor import (
    CombinedInvalidInputError,
    CombinedPredictor,
    CombinedResult,
)
from ml_engine.services.fake_news_predictor import PredictionResult
from ml_engine.services.sentiment_predictor import SentimentResult


def _fake_result(label='real', conf=0.8):
    return PredictionResult(
        label=label,
        confidence=conf,
        probabilities={
            'real': conf if label == 'real' else 1 - conf,
            'fake': conf if label == 'fake' else 1 - conf,
        },
        model_id='test-fake',
        model_name='Test Fake Model',
        algorithm='logistic_regression',
        feature_method='tfidf',
        processing_time_sec=0.01,
        analyzed_at='2026-01-01T00:00:00+00:00',
        text_preview='preview',
        headline='Headline',
    )


def _sent_result(label='positive', conf=0.7):
    probs = {'positive': 0.1, 'negative': 0.1, 'neutral': 0.1}
    probs[label] = conf
    return SentimentResult(
        label=label,
        confidence=conf,
        probabilities=probs,
        model_id='test-sent',
        model_name='Test Sentiment Model',
        algorithm='logistic_regression',
        feature_method='tfidf',
        processing_time_sec=0.01,
        analyzed_at='2026-01-01T00:00:00+00:00',
        text_preview='preview',
        headline='Headline',
    )


class CombinedPredictorUnitTests(TestCase):
    """Controlled combination coverage via mocks."""

    TEXT = 'This is a sufficiently long news article for combined analysis testing purposes.'

    def _run(self, fake_label, sent_label):
        with patch('ml_engine.services.combined_predictor.get_fake_predictor') as FP, \
             patch('ml_engine.services.combined_predictor.get_sentiment_predictor') as SP:
            fp_inst = MagicMock()
            fp_inst.predict.return_value = _fake_result(fake_label)
            FP.return_value = fp_inst
            sp_inst = MagicMock()
            sp_inst.predict.return_value = _sent_result(sent_label)
            SP.return_value = sp_inst
            return CombinedPredictor().analyze(self.TEXT, headline='Test')

    def test_all_six_combinations(self):
        pairs = [
            ('real', 'positive'),
            ('real', 'negative'),
            ('real', 'neutral'),
            ('fake', 'positive'),
            ('fake', 'negative'),
            ('fake', 'neutral'),
        ]
        for fake, sent in pairs:
            with self.subTest(fake=fake, sent=sent):
                result = self._run(fake, sent)
                self.assertTrue(result.both_succeeded)
                self.assertEqual(result.overall_fake_label, fake)
                self.assertEqual(result.overall_sentiment_label, sent)
                expected = f'{fake.capitalize()} + {sent.capitalize()}'
                self.assertEqual(result.combination, expected)
                self.assertIsNotNone(result.fake_news)
                self.assertIsNotNone(result.sentiment)
                self.assertIsNone(result.fake_error)
                self.assertIsNone(result.sentiment_error)

    def test_empty_input(self):
        with self.assertRaises(CombinedInvalidInputError):
            CombinedPredictor().analyze('')

    def test_short_input(self):
        with self.assertRaises(CombinedInvalidInputError):
            CombinedPredictor().analyze('short')

    def test_missing_fake_model_sentiment_ok(self):
        from ml_engine.services.fake_news_predictor import ModelLoadError

        with patch('ml_engine.services.combined_predictor.get_fake_predictor') as FP, \
             patch('ml_engine.services.combined_predictor.get_sentiment_predictor') as SP:
            FP.side_effect = ModelLoadError('No fake model')
            sp_inst = MagicMock()
            sp_inst.predict.return_value = _sent_result('neutral')
            SP.return_value = sp_inst

            result = CombinedPredictor().analyze(self.TEXT)
            self.assertIsNone(result.fake_news)
            self.assertIsNotNone(result.fake_error)
            self.assertIsNotNone(result.sentiment)
            self.assertEqual(result.overall_sentiment_label, 'neutral')
            self.assertTrue(result.has_any_result)
            self.assertFalse(result.both_succeeded)

    def test_missing_sentiment_model_fake_ok(self):
        from ml_engine.services.sentiment_predictor import SentimentModelLoadError

        with patch('ml_engine.services.combined_predictor.get_fake_predictor') as FP, \
             patch('ml_engine.services.combined_predictor.get_sentiment_predictor') as SP:
            fp_inst = MagicMock()
            fp_inst.predict.return_value = _fake_result('fake')
            FP.return_value = fp_inst
            SP.side_effect = SentimentModelLoadError('No sentiment model')

            result = CombinedPredictor().analyze(self.TEXT)
            self.assertIsNotNone(result.fake_news)
            self.assertIsNone(result.sentiment)
            self.assertIsNotNone(result.sentiment_error)
            self.assertEqual(result.overall_fake_label, 'fake')

    def test_both_models_unavailable(self):
        from ml_engine.services.fake_news_predictor import ModelLoadError
        from ml_engine.services.sentiment_predictor import SentimentModelLoadError

        with patch('ml_engine.services.combined_predictor.get_fake_predictor') as FP, \
             patch('ml_engine.services.combined_predictor.get_sentiment_predictor') as SP:
            FP.side_effect = ModelLoadError('No fake')
            SP.side_effect = SentimentModelLoadError('No sent')

            with self.assertRaises(CombinedInvalidInputError) as ctx:
                CombinedPredictor().analyze(self.TEXT)
            self.assertIn('Both models', str(ctx.exception))


class CombinedViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('combined_analysis:index')

    def test_get_page(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Combined Analysis')

    def test_empty_post(self):
        resp = self.client.post(self.url, {'action': 'analyse', 'content': '', 'headline': ''})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(Analysis.objects.filter(analysis_type='combined').count(), 0)

    def test_clear_action(self):
        resp = self.client.post(self.url, {
            'action': 'clear',
            'content': 'some text that is long enough',
            'headline': 'H',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context['form']['content'], '')

    @patch('combined_analysis.views.CombinedPredictor')
    def test_successful_save(self, MockPred):
        mock_result = CombinedResult(
            headline='Test Head',
            text_preview='preview',
            fake_news=_fake_result('real').to_dict(),
            sentiment=_sent_result('positive').to_dict(),
            overall_fake_label='real',
            overall_sentiment_label='positive',
            fake_model_name='FM',
            sentiment_model_name='SM',
            total_processing_time_sec=0.05,
            analyzed_at='2026-01-01T00:00:00+00:00',
            combination='Real + Positive',
        )
        inst = MagicMock()
        inst.analyze.return_value = mock_result
        MockPred.return_value = inst

        resp = self.client.post(self.url, {
            'action': 'analyse',
            'headline': 'Test Head',
            'content': 'This is a sufficiently long news article for testing database save behaviour.',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIsNotNone(resp.context.get('result'))
        self.assertEqual(Analysis.objects.filter(analysis_type='combined').count(), 1)
        rec = Analysis.objects.filter(analysis_type='combined').first()
        self.assertEqual(rec.fake_label, 'real')
        self.assertEqual(rec.sentiment_label, 'positive')
        self.assertEqual(rec.headline, 'Test Head')
