from django.contrib import messages
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from ml_engine.services.sentiment_predictor import (
    SentimentPredictor,
    SentimentModelLoadError,
    SentimentModelNotFoundError,
    SentimentInvalidInputError,
)
from ml_engine.services.model_registry import ModelRegistry
from ml_engine.services.predictor_cache import get_sentiment_predictor
from history.models import Analysis


@require_http_methods(['GET', 'POST'])
def analyze(request):
    registry = ModelRegistry()
    models = registry.list_models(task='sentiment')
    active = registry.get_active('sentiment')

    context = {
        'page_title': 'Sentiment Analysis',
        'active_menu': 'sentiment',
        'breadcrumbs': [{'name': 'Sentiment Analysis', 'url': None}],
        'models': models,
        'active_model': active,
        'result': None,
        'form': {
            'headline': '',
            'content': '',
            'model_id': active.model_id if active else '',
        },
    }

    if request.method == 'POST':
        headline = request.POST.get('headline', '').strip()
        content = request.POST.get('content', '').strip()
        model_id = request.POST.get('model_id', '').strip()
        clear = request.POST.get('action') == 'clear'

        uploaded = request.FILES.get('txt_file')
        if uploaded and uploaded.name.lower().endswith('.txt'):
            try:
                content = uploaded.read().decode('utf-8', errors='replace')
            except Exception:
                messages.error(request, 'Could not read the uploaded text file.')

        context['form'] = {
            'headline': headline,
            'content': content,
            'model_id': model_id,
        }

        if clear:
            context['form'] = {
                'headline': '', 'content': '',
                'model_id': active.model_id if active else '',
            }
            return render(request, 'sentiment/index.html', context)

        if not content and not headline:
            messages.error(request, 'Please provide a headline or news content.')
            return render(request, 'sentiment/index.html', context)

        try:
            predictor = get_sentiment_predictor(model_id or None)
            result = predictor.predict(content, headline=headline)
            context['result'] = result.to_dict()
            _save_analysis(result, content, headline)
        except (SentimentModelLoadError, SentimentModelNotFoundError) as e:
            messages.error(request, str(e))
        except SentimentInvalidInputError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f'Sentiment analysis failed: {e}')

    return render(request, 'sentiment/index.html', context)


def _save_analysis(result, content, headline):
    try:
        algo_map = {
            'naive_bayes': 'naive_bayes',
            'logistic_regression': 'logistic_regression',
            'word2vec_lr': 'word2vec_lr',
        }
        algo_key = algo_map.get(result.algorithm, 'other' if result.algorithm else '')
        Analysis.objects.create(
            text=content or headline,
            headline=headline[:300] if headline else (content[:100] if content else ''),
            analysis_type='sentiment',
            sentiment_label=result.label or '',
            sentiment_confidence=result.confidence,
            sentiment_model=algo_key,
            sentiment_model_name=getattr(result, 'model_name', '') or '',
            sentiment_model_id=getattr(result, 'model_id', '') or '',
            sentiment_scores=getattr(result, 'probabilities', None),
            processing_time=getattr(result, 'processing_time_sec', None),
        )
    except Exception:
        pass
