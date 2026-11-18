"""
Additional NLP helpers for sentiment analysis:
keyword extraction and word-frequency analysis.
These are frequency-based signals, not model explanations.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

from .text_preprocessing import PreprocessConfig, TextPreprocessor

# Simple polarity lexicons (frequency signals only — not model explanations)
POSITIVE_LEXICON = frozenset({
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
    'best', 'happy', 'positive', 'success', 'successful', 'win', 'won', 'gain',
    'improve', 'improved', 'growth', 'strong', 'progress', 'hope', 'promising',
    'outstanding', 'perfect', 'brilliant', 'joy', 'celebrate', 'victory',
    'benefit', 'advantage', 'effective', 'efficient', 'innovative', 'breakthrough',
})

NEGATIVE_LEXICON = frozenset({
    'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor', 'fail',
    'failed', 'failure', 'loss', 'lost', 'crisis', 'problem', 'issue', 'risk',
    'danger', 'threat', 'decline', 'drop', 'crash', 'fear', 'worry', 'concern',
    'negative', 'disaster', 'damage', 'harm', 'violence', 'death', 'killed',
    'scandal', 'corrupt', 'fraud', 'lie', 'false', 'weak', 'disappointing',
})


def extract_keywords(
    text: str,
    top_n: int = 15,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Frequency-based keyword extraction.
    Clearly separate from model explanations.
    """
    config = PreprocessConfig.for_sentiment()
    config.lemmatize = True
    config.stem = False
    pp = TextPreprocessor(config)
    result = pp.process(text or '')
    tokens = result.final_tokens

    freq = Counter(tokens)
    positive = [(w, c) for w, c in freq.most_common() if w in POSITIVE_LEXICON][:top_n]
    negative = [(w, c) for w, c in freq.most_common() if w in NEGATIVE_LEXICON][:top_n]
    frequent = freq.most_common(top_n)

    return {
        'positive_keywords': positive,
        'negative_keywords': negative,
        'frequent_terms': frequent,
        'note': (
            'Keyword lists are simple lexicon/frequency signals. '
            'They are not explanations of the machine learning model decision.'
        ),
    }
