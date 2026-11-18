"""
Feature extraction: TF-IDF and Word2Vec.
Fit only on training data to avoid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from gensim.models import Word2Vec
except ImportError:  # pragma: no cover
    Word2Vec = None


@dataclass
class TFIDFConfig:
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['ngram_range'] = list(self.ngram_range)
        return d


@dataclass
class Word2VecConfig:
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    workers: int = 2
    epochs: int = 10
    sg: int = 0  # 0=CBOW, 1=Skip-gram

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TFIDFExtractor:
    """
    TF-IDF feature extractor.
    Call fit() only on training texts, then transform train and test.
    """

    def __init__(self, config: Optional[TFIDFConfig] = None):
        self.config = config or TFIDFConfig()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._fitted = False

    def fit(self, texts: List[str]) -> 'TFIDFExtractor':
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            sublinear_tf=self.config.sublinear_tf,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]):
        if not self._fitted or self.vectorizer is None:
            raise RuntimeError('TFIDFExtractor must be fit on training data first.')
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]):
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocabulary_size(self) -> int:
        if self.vectorizer is None:
            return 0
        return len(self.vectorizer.vocabulary_)

    def get_feature_names(self) -> List[str]:
        if self.vectorizer is None:
            return []
        return list(self.vectorizer.get_feature_names_out())


class Word2VecExtractor:
    """
    Word2Vec document embeddings via mean pooling.
    Fit only on training tokenized documents.
    Handles empty / unknown-token documents safely (zero vector).
    """

    def __init__(self, config: Optional[Word2VecConfig] = None):
        if Word2Vec is None:
            raise ImportError('gensim is required for Word2VecExtractor')
        self.config = config or Word2VecConfig()
        self.model: Optional[Word2Vec] = None
        self._fitted = False

    @staticmethod
    def _tokenize(texts: List[str]) -> List[List[str]]:
        return [t.split() if isinstance(t, str) else [] for t in texts]

    def fit(self, texts: List[str]) -> 'Word2VecExtractor':
        sentences = self._tokenize(texts)
        # Filter empty
        sentences = [s for s in sentences if s]
        if not sentences:
            raise ValueError('No valid tokens found for Word2Vec training.')

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.config.vector_size,
            window=self.config.window,
            min_count=self.config.min_count,
            workers=self.config.workers,
            epochs=self.config.epochs,
            sg=self.config.sg,
        )
        self._fitted = True
        return self

    def _doc_vector(self, tokens: List[str]) -> np.ndarray:
        """Average word vectors; return zero vector if none available."""
        dim = self.config.vector_size
        if not self.model or not tokens:
            return np.zeros(dim, dtype=np.float32)

        vectors = []
        for w in tokens:
            if w in self.model.wv:
                vectors.append(self.model.wv[w])

        if not vectors:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32)

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted or self.model is None:
            raise RuntimeError('Word2VecExtractor must be fit on training data first.')
        tokenized = self._tokenize(texts)
        return np.vstack([self._doc_vector(toks) for toks in tokenized])

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocabulary_size(self) -> int:
        if self.model is None:
            return 0
        return len(self.model.wv)
