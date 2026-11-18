"""
Dataset loading, cleaning, and train/test splitting.
Fits feature extractors only on training data (no leakage).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_validation import (
    ColumnMapping,
    DatasetValidator,
    ValidationResult,
)
from .text_preprocessing import PreprocessConfig, TextPreprocessor


@dataclass
class LoadedDataset:
    """Container for a loaded and optionally preprocessed dataset."""

    name: str
    dataset_type: str
    df: pd.DataFrame                    # cleaned working frame
    mapping: ColumnMapping
    validation: ValidationResult
    original_row_count: int
    cleaned_row_count: int
    X_text: List[str] = field(default_factory=list)
    y: List[str] = field(default_factory=list)
    X_train: Optional[List[str]] = None
    X_test: Optional[List[str]] = None
    y_train: Optional[List[str]] = None
    y_test: Optional[List[str]] = None
    preprocess_config: Optional[Dict[str, Any]] = None
    vocabulary_size: int = 0
    status: str = 'loaded'              # loaded | preprocessed | split | error
    message: str = ''

    def to_summary(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'dataset_type': self.dataset_type,
            'original_row_count': self.original_row_count,
            'cleaned_row_count': self.cleaned_row_count,
            'class_distribution': self.validation.class_distribution,
            'missing_values': self.validation.missing_values,
            'duplicate_count': self.validation.duplicate_count,
            'vocabulary_size': self.vocabulary_size,
            'status': self.status,
            'message': self.message,
            'has_split': self.X_train is not None,
            'train_size': len(self.X_train) if self.X_train is not None else 0,
            'test_size': len(self.X_test) if self.X_test is not None else 0,
            'columns': self.validation.columns,
            'mapping': self.mapping.to_dict(),
        }


class DatasetLoader:
    """Load CSV datasets with flexible column mapping and optional preprocessing."""

    def __init__(
        self,
        dataset_type: str = 'fake_news',
        preprocess_config: Optional[PreprocessConfig] = None,
    ):
        self.dataset_type = dataset_type
        self.validator = DatasetValidator(dataset_type)
        if preprocess_config is None:
            if dataset_type == 'sentiment':
                preprocess_config = PreprocessConfig.for_sentiment()
            else:
                preprocess_config = PreprocessConfig.for_fake_news()
        self.preprocess_config = preprocess_config
        self.preprocessor = TextPreprocessor(preprocess_config)

    def read_csv(self, path_or_file, **kwargs) -> pd.DataFrame:
        """Read CSV with sensible defaults; raises ValueError on failure."""
        try:
            df = pd.read_csv(path_or_file, encoding='utf-8', **kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(path_or_file, encoding='latin-1', **kwargs)
        except Exception as e:
            raise ValueError(f'Failed to read CSV: {e}') from e

        if df.empty:
            raise ValueError('CSV file is empty.')
        return df

    def load(
        self,
        path_or_file,
        name: str = 'dataset',
        mapping: Optional[ColumnMapping] = None,
        drop_duplicates: bool = True,
        drop_missing: bool = True,
    ) -> LoadedDataset:
        """
        Load, validate, and clean a dataset.
        Does not preprocess text yet — call preprocess() separately.
        """
        df = self.read_csv(path_or_file)
        original_count = len(df)

        # Suggest mapping if not provided
        if mapping is None:
            suggested = self.validator.suggest_mapping(list(df.columns))
            mapping = ColumnMapping(
                text_column=suggested.get('text_column'),
                title_column=suggested.get('title_column'),
                label_column=suggested.get('label_column'),
                dataset_type=self.dataset_type,
            )
        else:
            mapping.dataset_type = self.dataset_type

        validation = self.validator.validate(df, mapping)

        if not validation.is_valid:
            return LoadedDataset(
                name=name,
                dataset_type=self.dataset_type,
                df=df,
                mapping=mapping,
                validation=validation,
                original_row_count=original_count,
                cleaned_row_count=0,
                status='error',
                message='; '.join(validation.errors),
            )

        work = df.copy()

        # Drop rows with missing text or label
        if drop_missing:
            subset = [c for c in [mapping.text_column, mapping.label_column] if c]
            work = work.dropna(subset=subset)

        # Normalize labels; drop rows with unknown labels
        labels_mapped, unmapped = self.validator.normalize_labels_series(
            work[mapping.label_column]
        )
        work = work.copy()
        work['_label'] = labels_mapped
        work = work.dropna(subset=['_label'])

        if drop_duplicates:
            # Deduplicate on text (+ title if present)
            dedup_cols = [mapping.text_column]
            if mapping.title_column:
                dedup_cols.append(mapping.title_column)
            work = work.drop_duplicates(subset=dedup_cols)

        # Build text field (optionally prepend title)
        texts = work[mapping.text_column].astype(str)
        if mapping.title_column and mapping.title_column in work.columns:
            titles = work[mapping.title_column].fillna('').astype(str)
            texts = (titles + ' ' + texts).str.strip()

        X_text = texts.tolist()
        y = work['_label'].tolist()

        # Update class distribution after cleaning
        from collections import Counter
        validation.class_distribution = dict(Counter(y))
        validation.row_count = len(work)

        return LoadedDataset(
            name=name,
            dataset_type=self.dataset_type,
            df=work.reset_index(drop=True),
            mapping=mapping,
            validation=validation,
            original_row_count=original_count,
            cleaned_row_count=len(work),
            X_text=X_text,
            y=y,
            status='loaded',
            message=f'Loaded {len(work)} rows from {original_count} original.',
        )

    def preprocess(self, dataset: LoadedDataset) -> LoadedDataset:
        """Apply text preprocessing to X_text. Updates vocabulary size."""
        if dataset.status == 'error':
            return dataset

        processed = self.preprocessor.transform_batch(dataset.X_text)
        # Drop empty after preprocessing
        pairs = [(t, label) for t, label in zip(processed, dataset.y) if t.strip()]
        if not pairs:
            dataset.status = 'error'
            dataset.message = 'All texts became empty after preprocessing.'
            return dataset

        X_clean, y_clean = zip(*pairs)
        dataset.X_text = list(X_clean)
        dataset.y = list(y_clean)
        dataset.cleaned_row_count = len(dataset.X_text)

        vocab = set()
        for t in dataset.X_text:
            vocab.update(t.split())
        dataset.vocabulary_size = len(vocab)
        dataset.preprocess_config = self.preprocess_config.to_dict()
        dataset.status = 'preprocessed'
        dataset.message = (
            f'Preprocessed {dataset.cleaned_row_count} texts. '
            f'Vocabulary size: {dataset.vocabulary_size}.'
        )
        return dataset

    def train_test_split(
        self,
        dataset: LoadedDataset,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> LoadedDataset:
        """
        Stratified train/test split.
        Reports if dataset is insufficient.
        """
        if dataset.status == 'error':
            return dataset

        if len(dataset.X_text) < 10:
            dataset.status = 'error'
            dataset.message = (
                f'Dataset too small for train/test split '
                f'({len(dataset.X_text)} rows). Need at least 10.'
            )
            return dataset

        stratify_labels = dataset.y if stratify else None
        # Check that every class has at least 2 samples for stratification
        if stratify:
            from collections import Counter
            counts = Counter(dataset.y)
            if any(c < 2 for c in counts.values()):
                stratify_labels = None
                dataset.message += ' Stratification disabled (some classes have <2 samples).'

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X_text,
            dataset.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels,
        )

        dataset.X_train = list(X_train)
        dataset.X_test = list(X_test)
        dataset.y_train = list(y_train)
        dataset.y_test = list(y_test)
        dataset.status = 'split'
        dataset.message = (
            f'Split: train={len(X_train)}, test={len(X_test)} '
            f'(test_size={test_size}, stratify={stratify_labels is not None}).'
        )
        return dataset
