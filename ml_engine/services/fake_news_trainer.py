"""
Train fake-news detection models: Multinomial NB, Logistic Regression (TF-IDF),
and Logistic Regression on Word2Vec document embeddings.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from .dataset_loader import DatasetLoader, LoadedDataset
from .dataset_validation import ColumnMapping
from .feature_extraction import TFIDFConfig, TFIDFExtractor, Word2VecConfig, Word2VecExtractor
from .model_evaluator import evaluate_classifier
from .model_registry import ModelMeta, ModelRegistry
from .text_preprocessing import PreprocessConfig


SUPPORTED_ALGORITHMS = {
    'naive_bayes': {
        'name': 'Multinomial Naïve Bayes',
        'feature_method': 'tfidf',
        'description': 'TF-IDF + MultinomialNB',
    },
    'logistic_regression': {
        'name': 'Logistic Regression (TF-IDF)',
        'feature_method': 'tfidf',
        'description': 'TF-IDF + Logistic Regression',
    },
    'word2vec_lr': {
        'name': 'Word2Vec + Logistic Regression',
        'feature_method': 'word2vec',
        'description': 'Averaged Word2Vec embeddings + Logistic Regression',
    },
}


class FakeNewsTrainer:
    """End-to-end training pipeline for fake news detection."""

    TASK = 'fake_news'

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.registry = ModelRegistry()

    def load_dataset(
        self,
        csv_path: str,
        dataset_name: str = 'dataset',
        dataset_id: Optional[int] = None,
        mapping: Optional[ColumnMapping] = None,
        test_size: float = 0.2,
    ) -> LoadedDataset:
        config = PreprocessConfig.for_fake_news()
        loader = DatasetLoader(dataset_type='fake_news', preprocess_config=config)
        loaded = loader.load(csv_path, name=dataset_name, mapping=mapping)
        if loaded.status == 'error':
            return loaded
        loaded = loader.preprocess(loaded)
        if loaded.status == 'error':
            return loaded
        loaded = loader.train_test_split(
            loaded, test_size=test_size, random_state=self.random_state, stratify=True
        )
        return loaded

    def _build_classifier(self, algorithm: str, **params):
        if algorithm == 'naive_bayes':
            alpha = float(params.get('alpha', 1.0))
            return MultinomialNB(alpha=alpha)
        if algorithm in ('logistic_regression', 'word2vec_lr'):
            C = float(params.get('C', 1.0))
            max_iter = int(params.get('max_iter', 500))
            return LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=self.random_state,
                solver='lbfgs',
            )
        raise ValueError(f'Unsupported algorithm: {algorithm}')

    def train(
        self,
        dataset: LoadedDataset,
        algorithm: str = 'logistic_regression',
        feature_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        set_active: bool = True,
    ) -> ModelMeta:
        """
        Train a model on an already-loaded & split dataset.
        Saves classifier, vectorizer/embeddings, and metadata to the registry.
        """
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f'Unknown algorithm "{algorithm}". '
                f'Choose from: {list(SUPPORTED_ALGORITHMS.keys())}'
            )

        if dataset.status not in ('split', 'preprocessed') or not dataset.X_train:
            raise ValueError(
                f'Dataset is not ready for training (status={dataset.status}). '
                'Load, preprocess, and split first.'
            )

        if len(dataset.X_train) < 5:
            raise ValueError(
                f'Training set too small ({len(dataset.X_train)} samples). '
                'Need more labelled data for reliable training.'
            )

        feature_params = feature_params or {}
        model_params = model_params or {}
        info = SUPPORTED_ALGORITHMS[algorithm]
        feature_method = info['feature_method']

        model_id = self.registry.create_id()
        display_name = model_name or f"{info['name']} · {dataset.name}"

        t0 = time.perf_counter()

        # --- Feature extraction (fit on TRAIN only) ---
        vectorizer = None
        w2v_extractor = None

        if feature_method == 'tfidf':
            tfidf_cfg = TFIDFConfig(
                max_features=int(feature_params.get('max_features', 5000)),
                ngram_range=tuple(feature_params.get('ngram_range', (1, 2))),
                min_df=int(feature_params.get('min_df', 1)),
                max_df=float(feature_params.get('max_df', 0.95)),
            )
            # For very small datasets, min_df=1 is safer
            if len(dataset.X_train) < 30:
                tfidf_cfg.min_df = 1
            vectorizer = TFIDFExtractor(tfidf_cfg)
            X_train = vectorizer.fit_transform(dataset.X_train)
            X_test = vectorizer.transform(dataset.X_test)
        else:
            w2v_cfg = Word2VecConfig(
                vector_size=int(feature_params.get('vector_size', 100)),
                window=int(feature_params.get('window', 5)),
                min_count=int(feature_params.get('min_count', 1)),
                epochs=int(feature_params.get('epochs', 20)),
            )
            w2v_extractor = Word2VecExtractor(w2v_cfg)
            X_train = w2v_extractor.fit_transform(dataset.X_train)
            X_test = w2v_extractor.transform(dataset.X_test)

        # --- Train classifier ---
        clf = self._build_classifier(algorithm, **model_params)
        clf.fit(X_train, dataset.y_train)
        train_time = time.perf_counter() - t0

        # --- Evaluate on held-out test set ---
        t1 = time.perf_counter()
        y_pred = clf.predict(X_test)
        test_time = time.perf_counter() - t1

        labels = sorted(set(dataset.y_train) | set(dataset.y_test))
        metrics = evaluate_classifier(dataset.y_test, list(y_pred), labels=labels)

        # --- Persist ---
        meta = ModelMeta(
            model_id=model_id,
            name=display_name,
            task=self.TASK,
            algorithm=algorithm,
            feature_method=feature_method,
            dataset_name=dataset.name,
            dataset_id=dataset_id,
            metrics=metrics,
            config={
                'algorithm': algorithm,
                'feature_method': feature_method,
                'feature_params': feature_params,
                'model_params': model_params,
                'random_state': self.random_state,
                'preprocess': dataset.preprocess_config or {},
                'labels': labels,
            },
            created_at=datetime.now(timezone.utc).isoformat(),
            is_active=False,
            train_size=len(dataset.X_train),
            test_size=len(dataset.X_test),
            training_time_sec=round(train_time, 4),
            testing_time_sec=round(test_time, 4),
            status='ready',
            message=(
                f'Trained on {len(dataset.X_train)} samples, '
                f'tested on {len(dataset.X_test)}. '
                f'Accuracy={metrics["accuracy"]:.4f}'
            ),
        )

        self.registry.register(meta)

        # Save artefacts
        joblib.dump(clf, meta.classifier_path)
        if vectorizer is not None:
            joblib.dump(vectorizer, meta.vectorizer_path)
        if w2v_extractor is not None and w2v_extractor.model is not None:
            w2v_extractor.model.save(str(meta.embedding_path))
            # Also save the extractor config wrapper for transform
            joblib.dump(
                {
                    'config': w2v_extractor.config.to_dict(),
                    'vector_size': w2v_extractor.config.vector_size,
                },
                meta.model_dir / 'w2v_meta.joblib',
            )

        if set_active:
            self.registry.set_active(model_id, self.TASK)

        return meta
