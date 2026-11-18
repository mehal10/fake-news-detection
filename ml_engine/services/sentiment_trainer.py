"""
Train 3-class sentiment classifiers: Logistic Regression and Multinomial NB on TF-IDF.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from .dataset_loader import DatasetLoader, LoadedDataset
from .dataset_validation import ColumnMapping
from .feature_extraction import TFIDFConfig, TFIDFExtractor
from .model_registry import ModelMeta, ModelRegistry
from .sentiment_evaluator import evaluate_sentiment
from .text_preprocessing import PreprocessConfig


SENTIMENT_ALGORITHMS = {
    'sentiment_lr': {
        'name': 'Logistic Regression (Sentiment)',
        'feature_method': 'tfidf',
        'description': 'TF-IDF + 3-class Logistic Regression',
    },
    'sentiment_nb': {
        'name': 'Naïve Bayes (Sentiment)',
        'feature_method': 'tfidf',
        'description': 'TF-IDF + MultinomialNB (3-class)',
    },
}


class SentimentTrainer:
    TASK = 'sentiment'

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
        config = PreprocessConfig.for_sentiment()  # preserve negation
        loader = DatasetLoader(dataset_type='sentiment', preprocess_config=config)
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

    def train(
        self,
        dataset: LoadedDataset,
        algorithm: str = 'sentiment_lr',
        feature_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        dataset_id: Optional[int] = None,
        set_active: bool = True,
    ) -> ModelMeta:
        if algorithm not in SENTIMENT_ALGORITHMS:
            raise ValueError(
                f'Unknown algorithm "{algorithm}". '
                f'Choose from: {list(SENTIMENT_ALGORITHMS.keys())}'
            )

        if not dataset.X_train or len(dataset.X_train) < 5:
            raise ValueError(
                f'Training set too small ({len(dataset.X_train) if dataset.X_train else 0}). '
                'Need more labelled sentiment data.'
            )

        feature_params = feature_params or {}
        model_params = model_params or {}
        info = SENTIMENT_ALGORITHMS[algorithm]

        model_id = self.registry.create_id()
        display_name = model_name or f"{info['name']} · {dataset.name}"

        t0 = time.perf_counter()

        tfidf_cfg = TFIDFConfig(
            max_features=int(feature_params.get('max_features', 5000)),
            ngram_range=tuple(feature_params.get('ngram_range', (1, 2))),
            min_df=int(feature_params.get('min_df', 1)),
            max_df=float(feature_params.get('max_df', 0.95)),
        )
        if len(dataset.X_train) < 30:
            tfidf_cfg.min_df = 1

        vectorizer = TFIDFExtractor(tfidf_cfg)
        X_train = vectorizer.fit_transform(dataset.X_train)
        X_test = vectorizer.transform(dataset.X_test)

        classes = np.unique(dataset.y_train)

        if algorithm == 'sentiment_nb':
            alpha = float(model_params.get('alpha', 1.0))
            clf = MultinomialNB(alpha=alpha)
        else:
            C = float(model_params.get('C', 1.0))
            max_iter = int(model_params.get('max_iter', 500))
            # Handle imbalance
            weights = compute_class_weight('balanced', classes=classes, y=dataset.y_train)
            class_weight = dict(zip(classes, weights))
            clf = LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=self.random_state,
                solver='lbfgs',
                # multi_class removed in sklearn 1.9+
                
                class_weight=class_weight,
            )

        clf.fit(X_train, dataset.y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = clf.predict(X_test)
        test_time = time.perf_counter() - t1

        labels = sorted(set(dataset.y_train) | set(dataset.y_test))
        metrics = evaluate_sentiment(list(dataset.y_test), list(y_pred), labels=labels)

        meta = ModelMeta(
            model_id=model_id,
            name=display_name,
            task=self.TASK,
            algorithm=algorithm,
            feature_method='tfidf',
            dataset_name=dataset.name,
            dataset_id=dataset_id,
            metrics=metrics,
            config={
                'algorithm': algorithm,
                'feature_method': 'tfidf',
                'feature_params': feature_params,
                'model_params': model_params,
                'random_state': self.random_state,
                'preprocess': dataset.preprocess_config or {},
                'labels': labels,
                'class_weight': 'balanced' if algorithm == 'sentiment_lr' else None,
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
                f'Accuracy={metrics["accuracy"]:.4f}, '
                f'Macro-F1={metrics["f1_macro"]:.4f}'
            ),
        )

        self.registry.register(meta)
        joblib.dump(clf, meta.classifier_path)
        joblib.dump(vectorizer, meta.vectorizer_path)

        if set_active:
            self.registry.set_active(model_id, self.TASK)

        return meta
