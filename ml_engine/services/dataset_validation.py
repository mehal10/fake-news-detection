"""
Dataset validation and label normalization for Fake News / Sentiment CSVs.
Flexible column mapping — does not assume fixed column names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Label maps — never assume unknown labels are real/neutral
# ---------------------------------------------------------------------------

FAKE_LABEL_MAP = {
    # real
    'real': 'real',
    'true': 'real',
    '0': 'real',
    0: 'real',
    'reliable': 'real',
    'authentic': 'real',
    'genuine': 'real',
    # fake
    'fake': 'fake',
    'false': 'fake',
    '1': 'fake',
    1: 'fake',
    'unreliable': 'fake',
    'hoax': 'fake',
    'rumor': 'fake',
    'rumour': 'fake',
}

SENTIMENT_LABEL_MAP = {
    # positive
    'positive': 'positive',
    'pos': 'positive',
    '1': 'positive',
    1: 'positive',
    'good': 'positive',
    'happy': 'positive',
    # negative
    'negative': 'negative',
    'neg': 'negative',
    '-1': 'negative',
    -1: 'negative',
    'bad': 'negative',
    'sad': 'negative',
    # neutral
    'neutral': 'neutral',
    'neu': 'neutral',
    '0': 'neutral',
    0: 'neutral',
    'mixed': 'neutral',
}


@dataclass
class ColumnMapping:
    """Flexible column mapping for uploaded datasets."""

    text_column: Optional[str] = None
    title_column: Optional[str] = None   # optional for fake news
    label_column: Optional[str] = None
    dataset_type: str = 'fake_news'  # 'fake_news' | 'sentiment'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    columns: List[str] = field(default_factory=list)
    missing_values: Dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    unmapped_labels: List[str] = field(default_factory=list)
    preview_rows: List[Dict[str, Any]] = field(default_factory=list)
    suggested_mapping: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetValidator:
    """Validate CSV datasets and normalize labels."""

    PREVIEW_LIMIT = 10
    MIN_ROWS_RECOMMENDED = 50

    # Common column name candidates (case-insensitive matching)
    TEXT_CANDIDATES = [
        'text', 'content', 'body', 'article', 'news', 'statement',
        'message', 'review', 'tweet', 'comment', 'description',
    ]
    TITLE_CANDIDATES = [
        'title', 'headline', 'subject', 'header',
    ]
    LABEL_CANDIDATES = [
        'label', 'class', 'target', 'category', 'sentiment',
        'polarity', 'fake', 'is_fake', 'type', 'verdict',
    ]

    def __init__(self, dataset_type: str = 'fake_news'):
        if dataset_type not in ('fake_news', 'sentiment'):
            raise ValueError("dataset_type must be 'fake_news' or 'sentiment'")
        self.dataset_type = dataset_type

    # ------------------------------------------------------------------
    # Column suggestion
    # ------------------------------------------------------------------

    def suggest_mapping(self, columns: List[str]) -> Dict[str, Optional[str]]:
        lower_map = {c.lower().strip(): c for c in columns}

        def pick(candidates: List[str]) -> Optional[str]:
            for cand in candidates:
                if cand in lower_map:
                    return lower_map[cand]
            # partial match
            for cand in candidates:
                for low, orig in lower_map.items():
                    if cand in low:
                        return orig
            return None

        return {
            'text_column': pick(self.TEXT_CANDIDATES),
            'title_column': pick(self.TITLE_CANDIDATES) if self.dataset_type == 'fake_news' else None,
            'label_column': pick(self.LABEL_CANDIDATES),
        }

    # ------------------------------------------------------------------
    # Label normalization
    # ------------------------------------------------------------------

    def normalize_label(self, raw) -> Optional[str]:
        """
        Map a raw label to canonical form.
        Returns None if the label is unknown (never assume real/neutral).
        """
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return None

        # Try exact / lowercased string match
        key = raw
        if isinstance(raw, str):
            key = raw.strip().lower()

        if self.dataset_type == 'fake_news':
            return FAKE_LABEL_MAP.get(key)
        return SENTIMENT_LABEL_MAP.get(key)

    def normalize_labels_series(self, series: pd.Series) -> Tuple[pd.Series, List[str]]:
        mapped = series.map(self.normalize_label)
        unmapped_mask = mapped.isna() & series.notna()
        unmapped = sorted(series[unmapped_mask].astype(str).unique().tolist())
        return mapped, unmapped

    # ------------------------------------------------------------------
    # Full validation
    # ------------------------------------------------------------------

    def validate(
        self,
        df: pd.DataFrame,
        mapping: Optional[ColumnMapping] = None,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []

        if df is None or not isinstance(df, pd.DataFrame):
            return ValidationResult(
                is_valid=False,
                errors=['Invalid data: expected a pandas DataFrame.'],
            )

        if df.empty:
            return ValidationResult(
                is_valid=False,
                errors=['Dataset is empty (0 rows).'],
                columns=list(df.columns),
            )

        columns = list(df.columns)
        suggested = self.suggest_mapping(columns)

        if mapping is None:
            mapping = ColumnMapping(
                text_column=suggested.get('text_column'),
                title_column=suggested.get('title_column'),
                label_column=suggested.get('label_column'),
                dataset_type=self.dataset_type,
            )

        # Required columns
        if not mapping.text_column:
            errors.append('Text column is required. Please map a text/content column.')
        elif mapping.text_column not in df.columns:
            errors.append(f'Text column "{mapping.text_column}" not found in dataset.')

        if not mapping.label_column:
            errors.append('Label column is required. Please map a label/class column.')
        elif mapping.label_column not in df.columns:
            errors.append(f'Label column "{mapping.label_column}" not found in dataset.')

        if mapping.title_column and mapping.title_column not in df.columns:
            warnings.append(f'Title column "{mapping.title_column}" not found; ignored.')
            mapping.title_column = None

        # Missing values
        missing = {}
        for col in columns:
            cnt = int(df[col].isna().sum())
            if cnt > 0:
                missing[col] = cnt

        if mapping.text_column and mapping.text_column in missing:
            warnings.append(
                f'Text column has {missing[mapping.text_column]} missing values; '
                'those rows will be dropped during loading.'
            )
        if mapping.label_column and mapping.label_column in missing:
            warnings.append(
                f'Label column has {missing[mapping.label_column]} missing values; '
                'those rows will be dropped during loading.'
            )

        # Duplicates (full-row)
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            warnings.append(f'{dup_count} duplicate row(s) detected.')

        # Class distribution (after normalization attempt)
        class_dist: Dict[str, int] = {}
        unmapped: List[str] = []
        if mapping.label_column and mapping.label_column in df.columns:
            mapped, unmapped = self.normalize_labels_series(df[mapping.label_column])
            class_dist = {
                str(k): int(v)
                for k, v in mapped.dropna().value_counts().items()
            }
            if unmapped:
                errors.append(
                    f'Unknown label values that cannot be mapped: {unmapped}. '
                    'Known fake-news labels: real/true/0, fake/false/1. '
                    'Known sentiment labels: positive/pos/1, negative/neg/-1, neutral/0. '
                    'Unknown labels are never assumed to be real or neutral.'
                )

        # Size warning
        if len(df) < self.MIN_ROWS_RECOMMENDED:
            warnings.append(
                f'Dataset has only {len(df)} rows. '
                f'At least {self.MIN_ROWS_RECOMMENDED} are recommended for reliable training.'
            )

        # Preview
        preview = df.head(self.PREVIEW_LIMIT).fillna('').astype(str).to_dict(orient='records')

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            column_count=len(columns),
            columns=columns,
            missing_values=missing,
            duplicate_count=dup_count,
            class_distribution=class_dist,
            unmapped_labels=unmapped,
            preview_rows=preview,
            suggested_mapping=suggested,
        )
