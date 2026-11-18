"""
Combined analysis: run fake-news and sentiment predictors on the same text.
Reuses existing predictors — no duplicated training or prediction logic.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .fake_news_predictor import (
    FakeNewsPredictor,
    InvalidInputError as FakeInvalidInputError,
    ModelLoadError as FakeModelLoadError,
    ModelNotFoundError as FakeModelNotFoundError,
    PredictionResult,
)
from .sentiment_predictor import (
    SentimentInvalidInputError,
    SentimentModelLoadError,
    SentimentModelNotFoundError,
    SentimentPredictor,
    SentimentResult,
)
from .predictor_cache import get_fake_predictor, get_sentiment_predictor


class CombinedInvalidInputError(Exception):
    pass


@dataclass
class CombinedResult:
    """Aggregated result of simultaneous fake-news + sentiment analysis."""

    headline: str
    text_preview: str
    # Fake news panel (None if model unavailable / failed)
    fake_news: Optional[Dict[str, Any]] = None
    fake_error: Optional[str] = None
    # Sentiment panel
    sentiment: Optional[Dict[str, Any]] = None
    sentiment_error: Optional[str] = None
    # Summary
    overall_fake_label: str = ''
    overall_sentiment_label: str = ''
    fake_model_name: str = ''
    sentiment_model_name: str = ''
    total_processing_time_sec: float = 0.0
    analyzed_at: str = ''
    combination: str = ''  # e.g. "Real + Positive"
    message: str = (
        'Machine learning predictions are estimates based on patterns learned '
        'from training data. Sentiment does not indicate news credibility, and '
        'Fake/Real labels are not factual verification of article content.'
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_any_result(self) -> bool:
        return self.fake_news is not None or self.sentiment is not None

    @property
    def both_succeeded(self) -> bool:
        return self.fake_news is not None and self.sentiment is not None


class CombinedPredictor:
    """
    Orchestrates fake-news and sentiment prediction independently.
    Missing one model does not block the other.
    """

    MIN_TEXT_LENGTH = 10

    def __init__(
        self,
        fake_model_id: Optional[str] = None,
        sentiment_model_id: Optional[str] = None,
    ):
        self.fake_model_id = fake_model_id
        self.sentiment_model_id = sentiment_model_id

    def analyze(self, text: str, headline: str = '') -> CombinedResult:
        combined = (headline + ' ' + text).strip() if headline else (text or '').strip()
        if not combined:
            raise CombinedInvalidInputError('Please provide a news headline or content text.')
        if len(combined) < self.MIN_TEXT_LENGTH:
            raise CombinedInvalidInputError(
                f'Text is too short ({len(combined)} characters). '
                f'Provide at least {self.MIN_TEXT_LENGTH} characters.'
            )

        t0 = time.perf_counter()
        result = CombinedResult(
            headline=headline or '',
            text_preview=combined[:240],
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )

        # --- Fake news (independent) ---
        try:
            fp = get_fake_predictor(self.fake_model_id)
            fake_res: PredictionResult = fp.predict(text, headline=headline)
            result.fake_news = fake_res.to_dict()
            result.overall_fake_label = fake_res.label
            result.fake_model_name = fake_res.model_name
        except (FakeModelLoadError, FakeModelNotFoundError) as e:
            result.fake_error = str(e)
        except FakeInvalidInputError as e:
            # Input already validated; treat as soft failure for this panel
            result.fake_error = str(e)
        except Exception as e:
            result.fake_error = f'Fake news prediction failed: {e}'

        # --- Sentiment (independent) ---
        try:
            sp = get_sentiment_predictor(self.sentiment_model_id)
            sent_res: SentimentResult = sp.predict(text, headline=headline)
            result.sentiment = sent_res.to_dict()
            result.overall_sentiment_label = sent_res.label
            result.sentiment_model_name = sent_res.model_name
        except (SentimentModelLoadError, SentimentModelNotFoundError) as e:
            result.sentiment_error = str(e)
        except SentimentInvalidInputError as e:
            result.sentiment_error = str(e)
        except Exception as e:
            result.sentiment_error = f'Sentiment prediction failed: {e}'

        result.total_processing_time_sec = round(time.perf_counter() - t0, 4)

        # Combination string only when both succeeded
        if result.both_succeeded:
            fake_disp = result.overall_fake_label.capitalize()
            sent_disp = result.overall_sentiment_label.capitalize()
            result.combination = f'{fake_disp} + {sent_disp}'
        elif result.overall_fake_label:
            result.combination = result.overall_fake_label.capitalize()
        elif result.overall_sentiment_label:
            result.combination = result.overall_sentiment_label.capitalize()

        if not result.has_any_result:
            # Both failed — surface a combined message
            parts = []
            if result.fake_error:
                parts.append(f'Fake news: {result.fake_error}')
            if result.sentiment_error:
                parts.append(f'Sentiment: {result.sentiment_error}')
            raise CombinedInvalidInputError(
                'Both models are unavailable. ' + ' | '.join(parts)
                if parts
                else 'No trained models are available for combined analysis.'
            )

        return result
