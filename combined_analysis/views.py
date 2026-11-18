"""
Combined News Analysis views.
Runs fake-news detection and sentiment analysis on the same article.
"""

import json

from django.contrib import messages
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from history.models import Analysis
from ml_engine.services.combined_predictor import (
    CombinedInvalidInputError,
    CombinedPredictor,
)
from ml_engine.services.model_registry import ModelRegistry


@require_http_methods(['GET', 'POST'])
def combined(request):
    registry = ModelRegistry()
    fake_models = registry.list_models(task='fake_news')
    sentiment_models = registry.list_models(task='sentiment')
    active_fake = registry.get_active('fake_news')
    active_sentiment = registry.get_active('sentiment')

    context = {
        'page_title': 'Combined Analysis',
        'active_menu': 'combined',
        'breadcrumbs': [
            {'name': 'Combined Analysis', 'url': None},
        ],
        'fake_models': fake_models,
        'sentiment_models': sentiment_models,
        'active_fake': active_fake,
        'active_sentiment': active_sentiment,
        'result': None,
        'saved_id': None,
        'form': {
            'headline': '',
            'content': '',
            'fake_model_id': active_fake.model_id if active_fake else '',
            'sentiment_model_id': active_sentiment.model_id if active_sentiment else '',
        },
    }

    if request.method == 'POST':
        headline = request.POST.get('headline', '').strip()
        content = request.POST.get('content', '').strip()
        fake_model_id = request.POST.get('fake_model_id', '').strip()
        sentiment_model_id = request.POST.get('sentiment_model_id', '').strip()
        action = request.POST.get('action', 'analyse')

        uploaded = request.FILES.get('txt_file')
        if uploaded and uploaded.name.lower().endswith('.txt'):
            try:
                content = uploaded.read().decode('utf-8', errors='replace')
            except Exception:
                messages.error(request, 'Could not read the uploaded text file.')

        context['form'] = {
            'headline': headline,
            'content': content,
            'fake_model_id': fake_model_id,
            'sentiment_model_id': sentiment_model_id,
        }

        if action == 'clear':
            context['form'] = {
                'headline': '',
                'content': '',
                'fake_model_id': active_fake.model_id if active_fake else '',
                'sentiment_model_id': active_sentiment.model_id if active_sentiment else '',
            }
            return render(request, 'combined_analysis/index.html', context)

        if not content and not headline:
            messages.error(request, 'Please provide a headline or news content.')
            return render(request, 'combined_analysis/index.html', context)

        try:
            predictor = CombinedPredictor(
                fake_model_id=fake_model_id or None,
                sentiment_model_id=sentiment_model_id or None,
            )
            result = predictor.analyze(content, headline=headline)
            result_dict = result.to_dict()
            context['result'] = result_dict
            # Chart-ready JSON (safe for JS)
            if result_dict.get('fake_news'):
                context['fake_probs_json'] = json.dumps(
                    result_dict['fake_news'].get('probabilities') or {}
                )
            if result_dict.get('sentiment'):
                context['sentiment_probs_json'] = json.dumps(
                    result_dict['sentiment'].get('probabilities') or {}
                )

            # Surface panel-level errors as messages (partial success still shown)
            if result.fake_error and not result.fake_news:
                messages.warning(
                    request,
                    f'Fake news model unavailable: {result.fake_error}',
                )
            if result.sentiment_error and not result.sentiment:
                messages.warning(
                    request,
                    f'Sentiment model unavailable: {result.sentiment_error}',
                )

            saved = _save_analysis(result, content, headline)
            if saved:
                context['saved_id'] = saved.id

        except CombinedInvalidInputError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f'Combined analysis failed: {e}')

    return render(request, 'combined_analysis/index.html', context)


def _save_analysis(result, content, headline):
    """Persist successful combined analysis as a single history record."""
    try:
        fake = result.fake_news or {}
        sent = result.sentiment or {}
        algo_map = {
            'naive_bayes': 'naive_bayes',
            'logistic_regression': 'logistic_regression',
            'word2vec_lr': 'word2vec_lr',
        }
        fake_algo = algo_map.get(fake.get('algorithm', ''), 'other' if fake.get('algorithm') else '')
        sent_algo = algo_map.get(sent.get('algorithm', ''), 'other' if sent.get('algorithm') else '')
        return Analysis.objects.create(
            text=content or headline,
            headline=(headline[:300] if headline else (content[:100] if content else '')),
            analysis_type='combined',
            fake_label=fake.get('label', '') or '',
            fake_confidence=fake.get('confidence'),
            fake_model=fake_algo,
            fake_model_name=fake.get('model_name', '') or '',
            fake_model_id=fake.get('model_id', '') or '',
            fake_scores=fake.get('probabilities'),
            sentiment_label=sent.get('label', '') or '',
            sentiment_confidence=sent.get('confidence'),
            sentiment_model=sent_algo,
            sentiment_model_name=sent.get('model_name', '') or '',
            sentiment_model_id=sent.get('model_id', '') or '',
            sentiment_scores=sent.get('probabilities'),
            processing_time=getattr(result, 'total_processing_time_sec', None),
        )
    except Exception:
        return None
