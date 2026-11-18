from django.db import models
from django.utils import timezone


class Analysis(models.Model):
    """
    Stores every successful analysis performed by the system.
    A combined analysis is stored as a single record (not double-counted).
    Failed analyses are never persisted.
    """

    ANALYSIS_TYPE_CHOICES = [
        ('fake_news', 'Fake News Detection'),
        ('sentiment', 'Sentiment Analysis'),
        ('combined', 'Combined Analysis'),
    ]

    FAKE_LABEL_CHOICES = [
        ('real', 'Real'),
        ('fake', 'Fake'),
        ('', 'N/A'),
    ]

    SENTIMENT_CHOICES = [
        ('positive', 'Positive'),
        ('negative', 'Negative'),
        ('neutral', 'Neutral'),
        ('', 'N/A'),
    ]

    MODEL_CHOICES = [
        ('naive_bayes', 'Naïve Bayes'),
        ('logistic_regression', 'Logistic Regression'),
        ('word2vec_lr', 'Word2Vec + LR'),
        ('random_forest', 'Random Forest'),
        ('svm', 'SVM'),
        ('other', 'Other'),
        ('', 'N/A'),
    ]

    # Core content
    text = models.TextField(help_text='Original news text submitted for analysis')
    headline = models.CharField(
        max_length=300,
        blank=True,
        help_text='Optional short headline / first line extracted from text',
    )

    # Analysis metadata
    analysis_type = models.CharField(
        max_length=20,
        choices=ANALYSIS_TYPE_CHOICES,
        db_index=True,
    )

    # Fake news results (populated for fake_news and combined)
    fake_label = models.CharField(
        max_length=10,
        choices=FAKE_LABEL_CHOICES,
        blank=True,
        default='',
        db_index=True,
    )
    fake_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text='Confidence score 0–1 for fake/real prediction',
    )
    fake_model = models.CharField(
        max_length=40,
        choices=MODEL_CHOICES,
        blank=True,
        default='',
        db_index=True,
    )
    fake_model_name = models.CharField(
        max_length=120,
        blank=True,
        default='',
        help_text='Human-readable model name / version label',
    )
    fake_model_id = models.CharField(
        max_length=64,
        blank=True,
        default='',
        help_text='Registry model_id if available',
    )
    fake_scores = models.JSONField(
        null=True,
        blank=True,
        help_text='Class probability scores e.g. {"real": 0.7, "fake": 0.3}',
    )

    # Sentiment results (populated for sentiment and combined)
    sentiment_label = models.CharField(
        max_length=10,
        choices=SENTIMENT_CHOICES,
        blank=True,
        default='',
        db_index=True,
    )
    sentiment_confidence = models.FloatField(
        null=True,
        blank=True,
        help_text='Confidence score 0–1 for sentiment prediction',
    )
    sentiment_model = models.CharField(
        max_length=40,
        choices=MODEL_CHOICES,
        blank=True,
        default='',
        db_index=True,
    )
    sentiment_model_name = models.CharField(
        max_length=120,
        blank=True,
        default='',
    )
    sentiment_model_id = models.CharField(
        max_length=64,
        blank=True,
        default='',
    )
    sentiment_scores = models.JSONField(
        null=True,
        blank=True,
        help_text='Class probability scores e.g. {"positive": 0.5, ...}',
    )

    # Timing
    processing_time = models.FloatField(
        null=True,
        blank=True,
        help_text='Total processing time in seconds',
    )

    # Timestamps
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis'
        verbose_name_plural = 'Analyses'
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['analysis_type', 'created_at']),
            models.Index(fields=['fake_label', 'created_at']),
            models.Index(fields=['sentiment_label', 'created_at']),
            models.Index(fields=['headline']),
        ]

    def __str__(self):
        preview = (self.headline or self.text[:60]).strip()
        return f"{self.get_analysis_type_display()} · {preview}"

    def short_text(self, length=80):
        source = self.headline or self.text
        source = source.strip().replace('\n', ' ')
        if len(source) <= length:
            return source
        return source[:length].rsplit(' ', 1)[0] + '…'

    @property
    def display_fake_label(self):
        return self.get_fake_label_display() if self.fake_label else '—'

    @property
    def display_sentiment_label(self):
        return self.get_sentiment_label_display() if self.sentiment_label else '—'

    @property
    def models_display(self):
        """Compact model label for table rows."""
        parts = []
        if self.fake_model_name:
            parts.append(self.fake_model_name)
        elif self.fake_model:
            parts.append(self.get_fake_model_display())
        if self.sentiment_model_name:
            parts.append(self.sentiment_model_name)
        elif self.sentiment_model:
            parts.append(self.get_sentiment_model_display())
        return ' · '.join(parts) if parts else '—'

    def to_export_dict(self):
        return {
            'id': self.id,
            'analysis_type': self.analysis_type,
            'analysis_type_display': self.get_analysis_type_display(),
            'headline': self.headline,
            'content': self.text,
            'fake_label': self.fake_label,
            'fake_confidence': self.fake_confidence,
            'fake_scores': self.fake_scores,
            'fake_model': self.fake_model,
            'fake_model_name': self.fake_model_name,
            'fake_model_id': self.fake_model_id,
            'sentiment_label': self.sentiment_label,
            'sentiment_confidence': self.sentiment_confidence,
            'sentiment_scores': self.sentiment_scores,
            'sentiment_model': self.sentiment_model,
            'sentiment_model_name': self.sentiment_model_name,
            'sentiment_model_id': self.sentiment_model_id,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }
