from django.contrib import admin
from .models import Dataset


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        'name', 'dataset_type', 'status', 'original_rows',
        'cleaned_rows', 'vocabulary_size', 'created_at',
    )
    list_filter = ('dataset_type', 'status')
    search_fields = ('name', 'original_filename')
    readonly_fields = (
        'original_rows', 'cleaned_rows', 'column_names',
        'class_distribution', 'missing_values', 'duplicate_count',
        'vocabulary_size', 'validation_errors', 'validation_warnings',
        'message', 'created_at', 'updated_at',
    )
