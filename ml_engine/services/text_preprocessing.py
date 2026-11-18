"""
Reusable text preprocessing pipeline for Fake News Detection and Sentiment Analysis.

Configurable steps so different models can use appropriate preprocessing.
Preserves negation words for sentiment analysis when configured.
"""

from __future__ import annotations

import re
import html
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEGATION_WORDS = frozenset({
    'not', 'no', 'never', 'neither', 'nor', 'none', 'nobody',
    'nothing', 'nowhere', 'hardly', 'scarcely', 'barely',
    "n't", "not", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
    "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
})

# Keep numbers by default (useful for news); strip only pure noise
URL_PATTERN = re.compile(
    r'https?://\S+|www\.\S+|\S+\.(com|org|net|edu|gov|io|co)/\S*',
    re.IGNORECASE,
)
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
# Keep alphanumeric, apostrophes (contractions/negation), and basic spaces
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-z0-9'\s]")
MULTI_SPACE_PATTERN = re.compile(r'\s+')
# Optional: collapse repeated characters (e.g. "soooo" -> "soo") lightly
# We do not aggressively remove numbers.


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""

    lowercase: bool = True
    remove_urls: bool = True
    remove_html: bool = True
    remove_special_chars: bool = True
    normalize_whitespace: bool = True
    tokenize: bool = True
    remove_stopwords: bool = True
    preserve_negation: bool = True  # Keep "not", "never", etc.
    stem: bool = False
    lemmatize: bool = True
    min_token_length: int = 1
    # Language for stopwords
    language: str = 'english'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessConfig':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def for_fake_news(cls) -> 'PreprocessConfig':
        """Recommended defaults for fake-news classification."""
        return cls(
            preserve_negation=False,
            stem=False,
            lemmatize=True,
            remove_stopwords=True,
        )

    @classmethod
    def for_sentiment(cls) -> 'PreprocessConfig':
        """Recommended defaults for sentiment analysis (keep negations)."""
        return cls(
            preserve_negation=True,
            stem=False,
            lemmatize=True,
            remove_stopwords=True,
        )


@dataclass
class PreprocessResult:
    """Structured output of the preprocessing pipeline."""

    original: str
    cleaned: str
    tokens: List[str]
    stemmed: List[str]
    lemmatized: List[str]
    final_tokens: List[str]
    word_count_before: int
    word_count_after: int
    steps_applied: List[str] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        return ' '.join(self.final_tokens)


class TextPreprocessor:
    """
    Configurable text preprocessing pipeline.

    Example
    -------
    >>> pp = TextPreprocessor(PreprocessConfig.for_sentiment())
    >>> result = pp.process("This is NOT a good product!!!")
    >>> result.final_tokens
    ['not', 'good', 'product']
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        self._stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        try:
            sw = set(stopwords.words(self.config.language))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            sw = set(stopwords.words(self.config.language))

        if self.config.preserve_negation:
            sw -= NEGATION_WORDS
        return sw

    def reload_stopwords(self) -> None:
        """Call after changing preserve_negation or language."""
        self._stopwords = self._load_stopwords()

    # ------------------------------------------------------------------
    # Individual steps (also usable independently)
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Apply cleaning steps without tokenization."""
        if text is None:
            return ''
        text = str(text)

        if self.config.remove_html:
            text = HTML_TAG_PATTERN.sub(' ', text)
            text = html.unescape(text)

        if self.config.remove_urls:
            text = URL_PATTERN.sub(' ', text)

        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_special_chars:
            text = SPECIAL_CHAR_PATTERN.sub(' ', text)

        if self.config.normalize_whitespace:
            text = MULTI_SPACE_PATTERN.sub(' ', text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        try:
            tokens = word_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            tokens = word_tokenize(text)
        return tokens

    def filter_tokens(self, tokens: List[str]) -> List[str]:
        result = []
        for t in tokens:
            if len(t) < self.config.min_token_length:
                continue
            if self.config.remove_stopwords and t in self._stopwords:
                continue
            result.append(t)
        return result

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self._stemmer.stem(t) for t in tokens]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        try:
            return [self._lemmatizer.lemmatize(t) for t in tokens]
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            return [self._lemmatizer.lemmatize(t) for t in tokens]

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process(self, text: str) -> PreprocessResult:
        """
        Run the full configurable pipeline and return a detailed result.
        Safe for empty / None input.
        """
        original = text if text is not None else ''
        steps: List[str] = []

        # Word count before (rough)
        word_count_before = len(str(original).split()) if original else 0

        cleaned = self.clean_text(original)
        if self.config.lowercase:
            steps.append('lowercase')
        if self.config.remove_urls:
            steps.append('remove_urls')
        if self.config.remove_html:
            steps.append('remove_html')
        if self.config.remove_special_chars:
            steps.append('remove_special_chars')
        if self.config.normalize_whitespace:
            steps.append('normalize_whitespace')

        tokens: List[str] = []
        if self.config.tokenize:
            tokens = self.tokenize(cleaned)
            steps.append('tokenize')
        else:
            tokens = cleaned.split() if cleaned else []

        filtered = self.filter_tokens(tokens)
        if self.config.remove_stopwords:
            steps.append('remove_stopwords')
        if self.config.preserve_negation:
            steps.append('preserve_negation')

        stemmed: List[str] = []
        lemmatized: List[str] = []
        final = filtered

        if self.config.stem:
            stemmed = self.stem_tokens(filtered)
            final = stemmed
            steps.append('stem')

        if self.config.lemmatize:
            # Lemmatize from filtered (or stemmed if both enabled — prefer lemma on filtered)
            base_for_lemma = filtered
            lemmatized = self.lemmatize_tokens(base_for_lemma)
            if not self.config.stem:
                final = lemmatized
            steps.append('lemmatize')

        # If both stem and lemmatize are on, final uses stemmed; still report both
        if self.config.stem and not stemmed:
            stemmed = self.stem_tokens(filtered)
        if self.config.lemmatize and not lemmatized:
            lemmatized = self.lemmatize_tokens(filtered)

        return PreprocessResult(
            original=original,
            cleaned=cleaned,
            tokens=tokens,
            stemmed=stemmed if stemmed else self.stem_tokens(filtered) if filtered else [],
            lemmatized=lemmatized if lemmatized else self.lemmatize_tokens(filtered) if filtered else [],
            final_tokens=final,
            word_count_before=word_count_before,
            word_count_after=len(final),
            steps_applied=steps,
        )

    def transform(self, text: str) -> str:
        """Convenience: return final cleaned text string only."""
        return self.process(text).final_text

    def transform_batch(self, texts: List[str]) -> List[str]:
        return [self.transform(t) for t in texts]


def preprocess_text(
    text: str,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """Module-level helper."""
    return TextPreprocessor(config).process(text)
