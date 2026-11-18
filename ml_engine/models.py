from django.db import models
from django.utils import timezone


class Dataset(models.Model):
    """Metadata for an uploaded training/evaluation dataset."""

    TYPE_CHOICES = [
        ('fake_news', 'Fake News'),
        ('sentiment', 'Sentiment'),
    ]

    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('validated', 'Validated'),
        ('invalid', 'Invalid'),
        ('preprocessed', 'Preprocessed'),
        ('ready', 'Ready for training'),
        ('error', 'Error'),
    ]

    name = models.CharField(max_length=200)
    dataset_type = models.CharField(max_length=20, choices=TYPE_CHOICES)
    file = models.FileField(upload_to='datasets/')
    original_filename = models.CharField(max_length=255, blank=True)

    # Column mapping (flexible)
    text_column = models.CharField(max_length=100, blank=True)
    title_column = models.CharField(max_length=100, blank=True)
    label_column = models.CharField(max_length=100, blank=True)

    # Stats
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    original_rows = models.PositiveIntegerField(default=0)
    cleaned_rows = models.PositiveIntegerField(default=0)
    column_names = models.JSONField(default=list, blank=True)
    class_distribution = models.JSONField(default=dict, blank=True)
    missing_values = models.JSONField(default=dict, blank=True)
    duplicate_count = models.PositiveIntegerField(default=0)
    vocabulary_size = models.PositiveIntegerField(default=0)
    validation_errors = models.JSONField(default=list, blank=True)
    validation_warnings = models.JSONField(default=list, blank=True)
    preprocess_config = models.JSONField(default=dict, blank=True)
    message = models.TextField(blank=True)

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.name} ({self.get_dataset_type_display()})'
