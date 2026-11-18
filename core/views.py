from django.shortcuts import render

from ml_engine.models import Dataset
from ml_engine.services.model_registry import ModelRegistry


def landing(request):
    context = {
        'page_title': 'Home',
        'active_menu': 'home',
    }
    return render(request, 'core/landing.html', context)


def about(request):
    """About Project — educational information and real system metadata."""
    datasets = list(
        Dataset.objects.filter(status__in=['ready', 'preprocessed', 'validated', 'uploaded'])
        .order_by('-updated_at')[:20]
    )

    registry = ModelRegistry()
    fake_models = registry.list_models(task='fake_news')
    sent_models = registry.list_models(task='sentiment')
    active_fake = registry.get_active('fake_news')
    active_sent = registry.get_active('sentiment')

    # Team placeholders — replace with real details when available
    team = {
        'members': [
            {'name': '[Team Member 1]', 'role': 'Developer'},
            {'name': '[Team Member 2]', 'role': 'Developer'},
            {'name': '[Team Member 3]', 'role': 'Developer'},
        ],
        'guide': '[Project Guide Name]',
        'college': '[College / University Name]',
        'department': '[Department Name]',
        'academic_year': '[Academic Year, e.g. 2025–26]',
    }

    context = {
        'page_title': 'About Project',
        'active_menu': 'about',
        'breadcrumbs': [
            {'name': 'About Project', 'url': None},
        ],
        'datasets': datasets,
        'fake_models': fake_models,
        'sent_models': sent_models,
        'active_fake': active_fake,
        'active_sent': active_sent,
        'team': team,
    }
    return render(request, 'core/about.html', context)
