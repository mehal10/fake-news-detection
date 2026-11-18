"""
Simple process-level cache for loaded predictors.
Avoids reloading Joblib/Gensim artifacts on every request in a single worker.
Cache is intentionally small and process-local (safe for runserver / single worker).
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, Optional, Tuple

_lock = Lock()
_fake_cache: Dict[str, object] = {}
_sentiment_cache: Dict[str, object] = {}
_MAX = 4


def get_fake_predictor(model_id: Optional[str] = None):
    from .fake_news_predictor import FakeNewsPredictor
    from .model_registry import ModelRegistry

    if not model_id:
        active = ModelRegistry().get_active("fake_news")
        model_id = active.model_id if active else None
    if not model_id:
        return FakeNewsPredictor()  # will raise on predict if none

    with _lock:
        if model_id in _fake_cache:
            return _fake_cache[model_id]
        pred = FakeNewsPredictor(model_id=model_id)
        if len(_fake_cache) >= _MAX:
            _fake_cache.pop(next(iter(_fake_cache)))
        _fake_cache[model_id] = pred
        return pred


def get_sentiment_predictor(model_id: Optional[str] = None):
    from .sentiment_predictor import SentimentPredictor
    from .model_registry import ModelRegistry

    if not model_id:
        active = ModelRegistry().get_active("sentiment")
        model_id = active.model_id if active else None
    if not model_id:
        return SentimentPredictor()

    with _lock:
        if model_id in _sentiment_cache:
            return _sentiment_cache[model_id]
        pred = SentimentPredictor(model_id=model_id)
        if len(_sentiment_cache) >= _MAX:
            _sentiment_cache.pop(next(iter(_sentiment_cache)))
        _sentiment_cache[model_id] = pred
        return pred


def clear_predictor_cache():
    with _lock:
        _fake_cache.clear()
        _sentiment_cache.clear()
