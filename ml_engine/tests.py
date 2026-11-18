from django.test import Client, TestCase
from django.urls import reverse

from ml_engine.services.text_preprocessing import PreprocessConfig, TextPreprocessor


class PreprocessServiceTests(TestCase):
    def test_empty(self):
        p = TextPreprocessor(PreprocessConfig.for_fake_news())
        out = p.process('')
        self.assertEqual(out.tokens, [])

    def test_url_and_html_removed(self):
        p = TextPreprocessor(PreprocessConfig.for_fake_news())
        out = p.process('Read <b>this</b> at https://example.com/news now')
        joined = ' '.join(out.tokens).lower()
        self.assertNotIn('http', joined)
        self.assertNotIn('<b>', joined)

    def test_negation_preserved_for_sentiment(self):
        p = TextPreprocessor(PreprocessConfig.for_sentiment())
        out = p.process('This is not good and never helpful')
        self.assertTrue(any(t in ('not', 'never') for t in out.tokens) or
                        any(t in ('not', 'never') for t in (out.lemmatized or out.stemmed or [])))

    def test_pages_load(self):
        c = Client()
        self.assertEqual(c.get(reverse('ml_engine:dataset_list')).status_code, 200)
        self.assertEqual(c.get(reverse('ml_engine:preprocess_demo')).status_code, 200)
        self.assertEqual(c.get(reverse('ml_engine:model_training')).status_code, 200)
