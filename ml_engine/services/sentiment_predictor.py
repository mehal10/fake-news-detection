"""
Sentiment prediction using trained 3-class models.
All results come from actual trained models — never hardcoded.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib

from .model_registry import ModelMeta, ModelRegistry
from .sentiment_services import extract_keywords
from .text_preprocessing import PreprocessConfig, TextPreprocessor


class SentimentModelNotFoundError(Exception):
    pass


class SentimentModelLoadError(Exception):
    pass


class SentimentInvalidInputError(Exception):
    pass


@dataclass
class SentimentResult:
    label: str  # positive | negative | neutral
    confidence: float
    probabilities: Dict[str, float]
    model_id: str
    model_name: str
    algorithm: str
    feature_method: str
    processing_time_sec: float
    analyzed_at: str
    text_preview: str
    headline: str = ''
    keywords: Dict[str, Any] = field(default_factory=dict)
    message: str = (
        'Sentiment scores are model estimates based on patterns in training data. '
        'They reflect linguistic polarity, not verified factual assessment of events.'
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SentimentPredictor:
    TASK = 'sentiment'
    MIN_TEXT_LENGTH = 10

    def __init__(self, model_id: Optional[str] = None):
        self.registry = ModelRegistry()
        self.meta: Optional[ModelMeta] = None
        self.classifier = None
        self.vectorizer = None
        self.preprocessor = TextPreprocessor(PreprocessConfig.for_sentiment())

        if model_id:
            self.load(model_id)
        else:
            active = self.registry.get_active(self.TASK)
            if active:
                self.load(active.model_id)

    def load(self, model_id: str) -> 'SentimentPredictor':
        meta = self.registry.get(model_id)
        if not meta:
            raise SentimentModelNotFoundError(f'Sentiment model "{model_id}" not found.')
        if meta.task != self.TASK:
            raise SentimentModelLoadError(f'Model "{model_id}" is not a sentiment model.')
        if meta.status != 'ready':
            raise SentimentModelLoadError(f'Model "{model_id}" is not ready.')

        if not meta.classifier_path.exists() or not meta.vectorizer_path.exists():
            raise SentimentModelLoadError(
                f'Model files missing for "{model_id}". Retrain the model.'
            )

        try:
            self.classifier = joblib.load(meta.classifier_path)
            self.vectorizer = joblib.load(meta.vectorizer_path)
        except Exception as e:
            raise SentimentModelLoadError(f'Failed to load model: {e}') from e

        self.meta = meta
        return self

    def predict(self, text: str, headline: str = '') -> SentimentResult:
        if self.meta is None or self.classifier is None or self.vectorizer is None:
            raise SentimentModelLoadError(
                'No trained sentiment model is available. '
                'Train a model from Model Training before analysing sentiment.'
            )

        combined = (headline + ' ' + text).strip() if headline else (text or '').strip()
        if not combined:
            raise SentimentInvalidInputError('Please provide news headline or content text.')
        if len(combined) < self.MIN_TEXT_LENGTH:
            raise SentimentInvalidInputError(
                f'Text is too short ({len(combined)} characters). '
                f'Provide at least {self.MIN_TEXT_LENGTH} characters.'
            )

        t0 = time.perf_counter()
        cleaned = self.preprocessor.transform(combined)
        if not cleaned.strip():
            raise SentimentInvalidInputError('Text became empty after preprocessing.')

        X = self.vectorizer.transform([cleaned])
        pred = str(self.classifier.predict(X)[0])

        probabilities: Dict[str, float] = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        confidence = 0.0
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(X)[0]
            classes = list(self.classifier.classes_)
            for cls, p in zip(classes, proba):
                probabilities[str(cls)] = round(float(p), 4)
            confidence = round(float(max(proba)), 4)
        else:
            probabilities[pred] = 1.0
            confidence = 1.0

        keywords = extract_keywords(combined)
        elapsed = time.perf_counter() - t0

        return SentimentResult(
            label=pred,
            confidence=confidence,
            probabilities=probabilities,
            model_id=self.meta.model_id,
            model_name=self.meta.name,
            algorithm=self.meta.algorithm,
            feature_method=self.meta.feature_method,
            processing_time_sec=round(elapsed, 4),
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            text_preview=combined[:200],
            headline=headline or '',
            keywords=keywords,
        )
