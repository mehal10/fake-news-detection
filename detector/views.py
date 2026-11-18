import time
from datetime import datetime

from django.contrib import messages
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods

from ml_engine.services.fake_news_predictor import (
    FakeNewsPredictor,
    ModelLoadError,
    ModelNotFoundError,
    InvalidInputError,
)
from ml_engine.services.model_registry import ModelRegistry
from ml_engine.services.predictor_cache import get_fake_predictor
from history.models import Analysis


@require_http_methods(['GET', 'POST'])
def detect(request):
    registry = ModelRegistry()
    models = registry.list_models(task='fake_news')
    active = registry.get_active('fake_news')

    context = {
        'page_title': 'Fake News Detection',
        'active_menu': 'detector',
        'breadcrumbs': [{'name': 'Fake News Detection', 'url': None}],
        'models': models,
        'active_model': active,
        'result': None,
        'compare_results': None,
        'form': {
            'headline': '',
            'content': '',
            'model_id': active.model_id if active else '',
            'compare': False,
        },
    }

    if request.method == 'POST':
        headline = request.POST.get('headline', '').strip()
        content = request.POST.get('content', '').strip()
        model_id = request.POST.get('model_id', '').strip()
        compare = request.POST.get('compare') == 'on'
        clear = request.POST.get('action') == 'clear'

        # TXT upload
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
            'compare': compare,
        }

        if clear:
            context['form'] = {
                'headline': '', 'content': '',
                'model_id': active.model_id if active else '',
                'compare': False,
            }
            return render(request, 'detector/index.html', context)

        if not content and not headline:
            messages.error(request, 'Please provide a headline or news content.')
            return render(request, 'detector/index.html', context)

        try:
            if compare:
                predictor = FakeNewsPredictor()
                results = predictor.predict_all_models(content, headline=headline)
                if not results:
                    messages.error(
                        request,
                        'No trained models available. Train a model first from Model Training.',
                    )
                else:
                    context['compare_results'] = [r.to_dict() for r in results]
                    # Save one analysis record using first result
                    _save_analysis(results[0], content, headline)
            else:
                predictor = get_fake_predictor(model_id or None)
                result = predictor.predict(content, headline=headline)
                context['result'] = result.to_dict()
                _save_analysis(result, content, headline)

        except (ModelLoadError, ModelNotFoundError) as e:
            messages.error(request, str(e))
        except InvalidInputError as e:
            messages.error(request, str(e))
        except Exception as e:
            messages.error(request, f'Prediction failed: {e}')

    return render(request, 'detector/index.html', context)


def _save_analysis(result, content, headline):
    """Persist successful prediction to Analysis history."""
    try:
        label = result.label if hasattr(result, 'label') else result.get('label', '')
        conf = result.confidence if hasattr(result, 'confidence') else result.get('confidence')
        algo = result.algorithm if hasattr(result, 'algorithm') else result.get('algorithm', '')
        probs = result.probabilities if hasattr(result, 'probabilities') else result.get('probabilities')
        model_name = result.model_name if hasattr(result, 'model_name') else result.get('model_name', '')
        model_id = result.model_id if hasattr(result, 'model_id') else result.get('model_id', '')
        proc = result.processing_time_sec if hasattr(result, 'processing_time_sec') else result.get('processing_time_sec')
        algo_map = {
            'naive_bayes': 'naive_bayes',
            'logistic_regression': 'logistic_regression',
            'word2vec_lr': 'word2vec_lr',
            'random_forest': 'random_forest',
            'svm': 'svm',
        }
        algo_key = algo_map.get(algo, 'other' if algo else '')
        Analysis.objects.create(
            text=content or headline,
            headline=headline[:300] if headline else (content[:100] if content else ''),
            analysis_type='fake_news',
            fake_label=label or '',
            fake_confidence=conf,
            fake_model=algo_key,
            fake_model_name=model_name or '',
            fake_model_id=model_id or '',
            fake_scores=probs,
            processing_time=proc,
        )
    except Exception:
        pass  # history save is best-effort
