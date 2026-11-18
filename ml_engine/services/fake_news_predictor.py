"""
Fake news prediction using trained models from the registry.
All predictions come from actual trained model files — never random/hardcoded.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from .feature_extraction import Word2VecConfig, Word2VecExtractor
from .model_registry import ModelMeta, ModelRegistry
from .text_preprocessing import PreprocessConfig, TextPreprocessor


class ModelNotFoundError(Exception):
    pass


class ModelLoadError(Exception):
    pass


class InvalidInputError(Exception):
    pass


@dataclass
class PredictionResult:
    label: str  # 'real' | 'fake'
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
    message: str = (
        'Machine learning predictions are estimates based on patterns learned '
        'from training data. They do not independently verify the factual '
        'accuracy of an article.'
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FakeNewsPredictor:
    """Load a trained model and run predictions."""

    TASK = 'fake_news'
    MIN_TEXT_LENGTH = 40

    def __init__(self, model_id: Optional[str] = None):
        self.registry = ModelRegistry()
        self.meta: Optional[ModelMeta] = None
        self.classifier = None
        self.vectorizer = None
        self.w2v_model = None
        self.w2v_config: Optional[Dict] = None
        self.preprocessor = TextPreprocessor(PreprocessConfig.for_fake_news())

        if model_id:
            self.load(model_id)
        else:
            active = self.registry.get_active(self.TASK)
            if active:
                self.load(active.model_id)

    def load(self, model_id: str) -> 'FakeNewsPredictor':
        meta = self.registry.get(model_id)
        if not meta:
            raise ModelNotFoundError(f'Model "{model_id}" not found in registry.')
        if meta.task != self.TASK:
            raise ModelLoadError(f'Model "{model_id}" is not a fake-news model.')
        if meta.status != 'ready':
            raise ModelLoadError(f'Model "{model_id}" is not ready (status={meta.status}).')

        if not meta.classifier_path.exists():
            raise ModelLoadError(
                f'Classifier file missing for model "{model_id}". '
                'Retrain the model or select another.'
            )

        try:
            self.classifier = joblib.load(meta.classifier_path)
        except Exception as e:
            raise ModelLoadError(f'Failed to load classifier: {e}') from e

        self.vectorizer = None
        self.w2v_model = None
        self.w2v_config = None

        if meta.feature_method == 'tfidf':
            if not meta.vectorizer_path.exists():
                raise ModelLoadError(f'Vectorizer file missing for model "{model_id}".')
            self.vectorizer = joblib.load(meta.vectorizer_path)
        elif meta.feature_method == 'word2vec':
            if not meta.embedding_path.exists():
                raise ModelLoadError(f'Word2Vec model file missing for "{model_id}".')
            try:
                from gensim.models import Word2Vec
                self.w2v_model = Word2Vec.load(str(meta.embedding_path))
            except Exception as e:
                raise ModelLoadError(f'Failed to load Word2Vec model: {e}') from e
            w2v_meta_path = meta.model_dir / 'w2v_meta.joblib'
            if w2v_meta_path.exists():
                self.w2v_config = joblib.load(w2v_meta_path)

        self.meta = meta
        return self

    def _transform(self, texts: List[str]):
        if self.meta is None or self.classifier is None:
            raise ModelLoadError('No model loaded. Train a model first.')

        if self.meta.feature_method == 'tfidf':
            if self.vectorizer is None:
                raise ModelLoadError('TF-IDF vectorizer not loaded.')
            return self.vectorizer.transform(texts)

        # Word2Vec mean pooling
        dim = 100
        if self.w2v_config:
            dim = int(self.w2v_config.get('vector_size', 100))
        elif self.w2v_model:
            dim = self.w2v_model.wv.vector_size

        vectors = []
        for text in texts:
            tokens = text.split()
            vecs = []
            if self.w2v_model:
                for w in tokens:
                    if w in self.w2v_model.wv:
                        vecs.append(self.w2v_model.wv[w])
            if vecs:
                vectors.append(np.mean(vecs, axis=0).astype(np.float32))
            else:
                vectors.append(np.zeros(dim, dtype=np.float32))
        return np.vstack(vectors)

    def predict(
        self,
        text: str,
        headline: str = '',
    ) -> PredictionResult:
        if self.meta is None or self.classifier is None:
            raise ModelLoadError(
                'No trained fake-news model is available. '
                'Please train a model from Dataset Management / Model Training before detecting.'
            )

        combined = (headline + ' ' + text).strip() if headline else (text or '').strip()
        if not combined:
            raise InvalidInputError('Please provide news headline or content text.')
        if len(combined) < self.MIN_TEXT_LENGTH:
            raise InvalidInputError(
                f'Text is too short ({len(combined)} characters). '
                f'Provide at least {self.MIN_TEXT_LENGTH} characters for a meaningful prediction.'
            )

        t0 = time.perf_counter()
        cleaned = self.preprocessor.transform(combined)
        if not cleaned.strip():
            raise InvalidInputError(
                'Text became empty after preprocessing. '
                'Try providing more substantive content.'
            )

        X = self._transform([cleaned])
        pred = self.classifier.predict(X)[0]
        label = str(pred)

        probabilities: Dict[str, float] = {}
        confidence = 0.0
        if hasattr(self.classifier, 'predict_proba'):
            proba = self.classifier.predict_proba(X)[0]
            classes = list(self.classifier.classes_)
            for cls, p in zip(classes, proba):
                probabilities[str(cls)] = round(float(p), 4)
            confidence = round(float(max(proba)), 4)
        else:
            probabilities[label] = 1.0
            confidence = 1.0

        elapsed = time.perf_counter() - t0

        msg = (
            'Machine learning predictions are estimates based on patterns learned '
            'from training data. They do not independently verify the factual '
            'accuracy of an article.'
        )
        if confidence < 0.6:
            msg = (
                'Low confidence prediction — the text may be outside the model\'s '
                'training domain (mainly English political news). Treat this result '
                'as uncertain. ' + msg
            )

        return PredictionResult(
            label=label,
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
            message=msg,
        )

    def predict_all_models(
        self,
        text: str,
        headline: str = '',
    ) -> List[PredictionResult]:
        """Run prediction with every ready fake-news model (compare mode)."""
        results = []
        models = self.registry.list_models(task=self.TASK)
        for meta in models:
            if meta.status != 'ready':
                continue
            try:
                predictor = FakeNewsPredictor(model_id=meta.model_id)
                results.append(predictor.predict(text, headline=headline))
            except Exception:
                continue
        return results
