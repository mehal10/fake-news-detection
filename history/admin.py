from django.contrib import admin
from .models import Analysis


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'analysis_type',
        'short_preview',
        'fake_label',
        'sentiment_label',
        'fake_model',
        'sentiment_model',
        'processing_time',
        'created_at',
    )
    list_filter = (
        'analysis_type',
        'fake_label',
        'sentiment_label',
        'fake_model',
        'sentiment_model',
        'created_at',
    )
    search_fields = ('text', 'headline', 'fake_model_name', 'sentiment_model_name', 'id')
    readonly_fields = (
        'created_at',
        'updated_at',
        'fake_scores',
        'sentiment_scores',
        'fake_model_id',
        'sentiment_model_id',
    )
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

    def short_preview(self, obj):
        return obj.short_text(50)

    short_preview.short_description = 'Preview'
