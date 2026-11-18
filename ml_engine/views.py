import json
import os
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.core.files.storage import default_storage
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods, require_POST, require_GET
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import Dataset
from .services.dataset_loader import DatasetLoader
from .services.dataset_validation import ColumnMapping, DatasetValidator
from .services.text_preprocessing import (
    TextPreprocessor,
    PreprocessConfig,
    preprocess_text,
)


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

def dataset_list(request):
    datasets = Dataset.objects.all()
    context = {
        'page_title': 'Dataset Management',
        'active_menu': 'datasets',
        'breadcrumbs': [{'name': 'Dataset Management', 'url': None}],
        'datasets': datasets,
    }
    return render(request, 'ml_engine/datasets.html', context)


@require_http_methods(['GET', 'POST'])
def dataset_upload(request):
    context = {
        'page_title': 'Upload Dataset',
        'active_menu': 'datasets',
        'breadcrumbs': [
            {'name': 'Dataset Management', 'url': '/ml/datasets/'},
            {'name': 'Upload', 'url': None},
        ],
    }

    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        dataset_type = request.POST.get('dataset_type', 'fake_news')
        uploaded = request.FILES.get('file')
        text_col = request.POST.get('text_column', '').strip() or None
        title_col = request.POST.get('title_column', '').strip() or None
        label_col = request.POST.get('label_column', '').strip() or None

        if not name:
            messages.error(request, 'Dataset name is required.')
            return render(request, 'ml_engine/dataset_upload.html', context)

        if not uploaded:
            messages.error(request, 'Please select a CSV file.')
            return render(request, 'ml_engine/dataset_upload.html', context)

        if not uploaded.name.lower().endswith('.csv'):
            messages.error(request, 'Only CSV files are accepted.')
            return render(request, 'ml_engine/dataset_upload.html', context)

        # Reject oversized uploads (also limited by DATA_UPLOAD_MAX_MEMORY_SIZE)
        max_bytes = getattr(settings, 'MAX_DATASET_UPLOAD_BYTES', 25 * 1024 * 1024)
        if uploaded.size and uploaded.size > max_bytes:
            messages.error(
                request,
                f'File is too large ({uploaded.size // (1024 * 1024)} MB). '
                f'Maximum allowed size is {max_bytes // (1024 * 1024)} MB.',
            )
            return render(request, 'ml_engine/dataset_upload.html', context)

        # Basic content sniff: first bytes should look like text/CSV, not binary
        try:
            head = uploaded.read(512)
            uploaded.seek(0)
            if b'\x00' in head:
                messages.error(request, 'File appears to be binary, not a text CSV.')
                return render(request, 'ml_engine/dataset_upload.html', context)
        except Exception:
            uploaded.seek(0)

        if dataset_type not in ('fake_news', 'sentiment'):
            messages.error(request, 'Invalid dataset type.')
            return render(request, 'ml_engine/dataset_upload.html', context)

        # Save file first
        ds = Dataset(
            name=name,
            dataset_type=dataset_type,
            file=uploaded,
            original_filename=uploaded.name,
            text_column=text_col or '',
            title_column=title_col or '',
            label_column=label_col or '',
            status='uploaded',
        )
        ds.save()

        # Validate
        try:
            loader = DatasetLoader(dataset_type=dataset_type)
            mapping = ColumnMapping(
                text_column=text_col,
                title_column=title_col,
                label_column=label_col,
                dataset_type=dataset_type,
            )
            # If mapping incomplete, loader will use suggestions
            if not text_col and not label_col:
                mapping = None

            loaded = loader.load(ds.file.path, name=name, mapping=mapping)

            ds.original_rows = loaded.original_row_count
            ds.cleaned_rows = loaded.cleaned_row_count
            ds.column_names = loaded.validation.columns
            ds.class_distribution = loaded.validation.class_distribution
            ds.missing_values = loaded.validation.missing_values
            ds.duplicate_count = loaded.validation.duplicate_count
            ds.validation_errors = loaded.validation.errors
            ds.validation_warnings = loaded.validation.warnings
            ds.message = loaded.message

            # Persist suggested/used mapping
            if loaded.mapping.text_column:
                ds.text_column = loaded.mapping.text_column
            if loaded.mapping.title_column:
                ds.title_column = loaded.mapping.title_column
            if loaded.mapping.label_column:
                ds.label_column = loaded.mapping.label_column

            if loaded.status == 'error':
                ds.status = 'invalid'
                messages.error(request, f'Dataset invalid: {loaded.message}')
            else:
                ds.status = 'validated'
                messages.success(
                    request,
                    f'Dataset "{name}" uploaded and validated '
                    f'({loaded.cleaned_row_count} clean rows).',
                )
            ds.save()
        except Exception as e:
            ds.status = 'error'
            ds.message = str(e)
            ds.validation_errors = [str(e)]
            ds.save()
            messages.error(request, 'Failed to process dataset. Ensure the file is a valid UTF-8 CSV with a header row.')

        return redirect('ml_engine:dataset_detail', pk=ds.pk)

    return render(request, 'ml_engine/dataset_upload.html', context)


def dataset_detail(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    preview = []
    suggested = {}

    if ds.file and os.path.exists(ds.file.path):
        try:
            loader = DatasetLoader(dataset_type=ds.dataset_type)
            mapping = ColumnMapping(
                text_column=ds.text_column or None,
                title_column=ds.title_column or None,
                label_column=ds.label_column or None,
                dataset_type=ds.dataset_type,
            )
            df = loader.read_csv(ds.file.path)
            validator = DatasetValidator(ds.dataset_type)
            result = validator.validate(df, mapping if ds.text_column else None)
            # Ordered preview values matching column order
            cols = result.columns
            preview = [
                [str(row.get(c, '')) for c in cols]
                for row in result.preview_rows
            ]
            suggested = result.suggested_mapping
        except Exception as e:
            messages.warning(request, f'Could not load preview: {e}')

    context = {
        'page_title': f'Dataset: {ds.name}',
        'active_menu': 'datasets',
        'breadcrumbs': [
            {'name': 'Dataset Management', 'url': '/ml/datasets/'},
            {'name': ds.name, 'url': None},
        ],
        'dataset': ds,
        'preview': preview,
        'suggested': suggested,
    }
    return render(request, 'ml_engine/dataset_detail.html', context)


@require_POST
def dataset_preprocess(request, pk):
    """Run preprocessing on a validated dataset and update stats."""
    ds = get_object_or_404(Dataset, pk=pk)

    if ds.status in ('invalid', 'error', 'uploaded'):
        messages.error(request, 'Dataset must be validated before preprocessing.')
        return redirect('ml_engine:dataset_detail', pk=pk)

    try:
        if ds.dataset_type == 'sentiment':
            config = PreprocessConfig.for_sentiment()
        else:
            config = PreprocessConfig.for_fake_news()

        loader = DatasetLoader(dataset_type=ds.dataset_type, preprocess_config=config)
        mapping = ColumnMapping(
            text_column=ds.text_column or None,
            title_column=ds.title_column or None,
            label_column=ds.label_column or None,
            dataset_type=ds.dataset_type,
        )
        loaded = loader.load(ds.file.path, name=ds.name, mapping=mapping)
        if loaded.status == 'error':
            messages.error(request, loaded.message)
            return redirect('ml_engine:dataset_detail', pk=pk)

        loaded = loader.preprocess(loaded)
        loaded = loader.train_test_split(loaded)

        ds.cleaned_rows = loaded.cleaned_row_count
        ds.vocabulary_size = loaded.vocabulary_size
        ds.class_distribution = loaded.validation.class_distribution
        ds.preprocess_config = loaded.preprocess_config or config.to_dict()
        ds.status = 'ready' if loaded.status == 'split' else loaded.status
        ds.message = loaded.message
        ds.save()

        messages.success(
            request,
            f'Preprocessing complete. Vocab size: {ds.vocabulary_size}. {loaded.message}',
        )
    except Exception as e:
        ds.status = 'error'
        ds.message = str(e)
        ds.save()
        messages.error(request, f'Preprocessing failed: {e}')

    return redirect('ml_engine:dataset_detail', pk=pk)


@require_POST
def dataset_delete(request, pk):
    ds = get_object_or_404(Dataset, pk=pk)
    name = ds.name
    if ds.file:
        ds.file.delete(save=False)
    ds.delete()
    messages.success(request, f'Dataset "{name}" deleted.')
    return redirect('ml_engine:dataset_list')


@require_POST
def dataset_remap(request, pk):
    """Update column mapping and re-validate."""
    ds = get_object_or_404(Dataset, pk=pk)
    text_col = request.POST.get('text_column', '').strip()
    title_col = request.POST.get('title_column', '').strip()
    label_col = request.POST.get('label_column', '').strip()

    ds.text_column = text_col
    ds.title_column = title_col
    ds.label_column = label_col
    ds.save()

    try:
        loader = DatasetLoader(dataset_type=ds.dataset_type)
        mapping = ColumnMapping(
            text_column=text_col or None,
            title_column=title_col or None,
            label_column=label_col or None,
            dataset_type=ds.dataset_type,
        )
        loaded = loader.load(ds.file.path, name=ds.name, mapping=mapping)

        ds.original_rows = loaded.original_row_count
        ds.cleaned_rows = loaded.cleaned_row_count
        ds.column_names = loaded.validation.columns
        ds.class_distribution = loaded.validation.class_distribution
        ds.missing_values = loaded.validation.missing_values
        ds.duplicate_count = loaded.validation.duplicate_count
        ds.validation_errors = loaded.validation.errors
        ds.validation_warnings = loaded.validation.warnings
        ds.message = loaded.message
        ds.status = 'validated' if loaded.status != 'error' else 'invalid'
        if loaded.mapping.text_column:
            ds.text_column = loaded.mapping.text_column
        if loaded.mapping.label_column:
            ds.label_column = loaded.mapping.label_column
        ds.save()

        if ds.status == 'validated':
            messages.success(request, 'Column mapping updated and dataset re-validated.')
        else:
            messages.error(request, f'Validation failed: {loaded.message}')
    except Exception as e:
        messages.error(request, str(e))

    return redirect('ml_engine:dataset_detail', pk=pk)


# ---------------------------------------------------------------------------
# Preprocessing demonstration
# ---------------------------------------------------------------------------

@ensure_csrf_cookie
def preprocess_demo(request):
    context = {
        'page_title': 'Text Preprocessing Demo',
        'active_menu': 'preprocess',
        'breadcrumbs': [{'name': 'Text Preprocessing', 'url': None}],
        'result': None,
        'input_text': '',
        'config': PreprocessConfig.for_sentiment().to_dict(),
    }
    return render(request, 'ml_engine/preprocess_demo.html', context)


@require_POST
def preprocess_api(request):
    """AJAX endpoint for interactive preprocessing demo."""
    try:
        body = json.loads(request.body.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError):
        body = request.POST

    text = body.get('text', '')
    cfg_data = body.get('config') or {}

    config = PreprocessConfig(
        lowercase=bool(cfg_data.get('lowercase', True)),
        remove_urls=bool(cfg_data.get('remove_urls', True)),
        remove_html=bool(cfg_data.get('remove_html', True)),
        remove_special_chars=bool(cfg_data.get('remove_special_chars', True)),
        normalize_whitespace=bool(cfg_data.get('normalize_whitespace', True)),
        tokenize=bool(cfg_data.get('tokenize', True)),
        remove_stopwords=bool(cfg_data.get('remove_stopwords', True)),
        preserve_negation=bool(cfg_data.get('preserve_negation', True)),
        stem=bool(cfg_data.get('stem', False)),
        lemmatize=bool(cfg_data.get('lemmatize', True)),
        min_token_length=int(cfg_data.get('min_token_length', 1)),
    )

    result = TextPreprocessor(config).process(text)

    return JsonResponse({
        'original': result.original,
        'cleaned': result.cleaned,
        'tokens': result.tokens,
        'stemmed': result.stemmed,
        'lemmatized': result.lemmatized,
        'final_tokens': result.final_tokens,
        'final_text': result.final_text,
        'word_count_before': result.word_count_before,
        'word_count_after': result.word_count_after,
        'steps_applied': result.steps_applied,
    })


# ---------------------------------------------------------------------------
# Model training (Fake News)
# ---------------------------------------------------------------------------

def model_training(request):
    from .services.model_registry import ModelRegistry
    from .services.fake_news_trainer import SUPPORTED_ALGORITHMS
    from .services.sentiment_trainer import SENTIMENT_ALGORITHMS

    registry = ModelRegistry()
    task = request.GET.get('task', 'fake_news')
    if task not in ('fake_news', 'sentiment'):
        task = 'fake_news'

    models = registry.list_models(task=task)
    datasets = Dataset.objects.filter(
        dataset_type=task,
        status__in=['validated', 'preprocessed', 'ready'],
    )
    algorithms = SUPPORTED_ALGORITHMS if task == 'fake_news' else SENTIMENT_ALGORITHMS

    context = {
        'page_title': 'Model Training',
        'active_menu': 'training',
        'breadcrumbs': [{'name': 'Model Training', 'url': None}],
        'datasets': datasets,
        'algorithms': algorithms,
        'models': models,
        'active_model': registry.get_active(task),
        'task': task,
        'fake_models': registry.list_models(task='fake_news'),
        'sentiment_models': registry.list_models(task='sentiment'),
        'active_fake': registry.get_active('fake_news'),
        'active_sentiment': registry.get_active('sentiment'),
    }
    return render(request, 'ml_engine/model_training.html', context)


@require_POST
def model_train_start(request):
    """Start model training (fake news or sentiment) from a selected dataset."""
    from .services.fake_news_trainer import FakeNewsTrainer, SUPPORTED_ALGORITHMS
    from .services.sentiment_trainer import SentimentTrainer, SENTIMENT_ALGORITHMS
    from .services.dataset_validation import ColumnMapping

    dataset_id = request.POST.get('dataset_id')
    algorithm = request.POST.get('algorithm', 'logistic_regression')
    test_size = float(request.POST.get('test_size', 0.2))
    model_name = request.POST.get('model_name', '').strip()
    max_features = int(request.POST.get('max_features', 5000))
    alpha = float(request.POST.get('alpha', 1.0))
    C = float(request.POST.get('C', 1.0))
    task = request.POST.get('task', 'fake_news')

    is_sentiment = algorithm in SENTIMENT_ALGORITHMS or task == 'sentiment'
    if is_sentiment:
        if algorithm not in SENTIMENT_ALGORITHMS:
            algorithm = 'sentiment_lr'
        allowed = SENTIMENT_ALGORITHMS
        task = 'sentiment'
    else:
        if algorithm not in SUPPORTED_ALGORITHMS:
            messages.error(request, f'Unsupported algorithm: {algorithm}')
            return redirect('ml_engine:model_training')
        allowed = SUPPORTED_ALGORITHMS
        task = 'fake_news'

    try:
        ds = Dataset.objects.get(pk=dataset_id, dataset_type=task)
    except Dataset.DoesNotExist:
        messages.error(request, f'Selected dataset not found or not a {task} dataset.')
        return redirect(f'/ml/training/?task={task}')

    if not ds.file or not os.path.exists(ds.file.path):
        messages.error(request, 'Dataset file is missing on disk.')
        return redirect(f'/ml/training/?task={task}')

    try:
        mapping = ColumnMapping(
            text_column=ds.text_column or None,
            title_column=ds.title_column or None,
            label_column=ds.label_column or None,
            dataset_type=task,
        )
        feature_params = {
            'max_features': max_features,
            'ngram_range': (1, 2),
            'min_df': 1,
            'vector_size': 100,
            'min_count': 1,
            'epochs': 30,
        }
        model_params = {'alpha': alpha, 'C': C, 'max_iter': 500}

        if is_sentiment:
            trainer = SentimentTrainer(random_state=42)
        else:
            trainer = FakeNewsTrainer(random_state=42)

        loaded = trainer.load_dataset(
            ds.file.path,
            dataset_name=ds.name,
            dataset_id=ds.pk,
            mapping=mapping,
            test_size=test_size,
        )
        if loaded.status == 'error':
            messages.error(request, f'Dataset error: {loaded.message}')
            return redirect(f'/ml/training/?task={task}')

        if loaded.cleaned_row_count >= 30:
            feature_params['min_df'] = 2

        meta = trainer.train(
            loaded,
            algorithm=algorithm,
            feature_params=feature_params,
            model_params=model_params,
            model_name=model_name or None,
            dataset_id=ds.pk,
            set_active=True,
        )

        ds.status = 'ready'
        ds.vocabulary_size = loaded.vocabulary_size
        ds.cleaned_rows = loaded.cleaned_row_count
        ds.message = meta.message
        ds.save()

        messages.success(
            request,
            f'Model trained successfully: {meta.name}. '
            f'Accuracy={meta.metrics.get("accuracy", "—")} '
            f'(train={meta.train_size}, test={meta.test_size}).',
        )
        return redirect('ml_engine:model_detail', model_id=meta.model_id)

    except Exception as e:
        messages.error(request, f'Training failed: {e}')
        return redirect(f'/ml/training/?task={task}')


def model_detail(request, model_id):
    from .services.model_registry import ModelRegistry

    registry = ModelRegistry()
    meta = registry.get(model_id)
    if not meta:
        messages.error(request, 'Model not found.')
        return redirect('ml_engine:model_training')

    labels = meta.metrics.get('labels', [])
    cm = meta.metrics.get('confusion_matrix', [])
    cm_rows = list(zip(labels, cm)) if cm and labels else []
    context = {
        'page_title': f'Model: {meta.name}',
        'active_menu': 'training',
        'breadcrumbs': [
            {'name': 'Model Training', 'url': '/ml/training/'},
            {'name': meta.name, 'url': None},
        ],
        'meta': meta,
        'cm_labels': labels,
        'cm': cm,
        'cm_rows': cm_rows,
    }
    return render(request, 'ml_engine/model_detail.html', context)


@require_POST
def model_set_active(request, model_id):
    from .services.model_registry import ModelRegistry

    registry = ModelRegistry()
    meta = registry.get(model_id)
    if not meta:
        messages.error(request, 'Model not found.')
        return redirect('ml_engine:model_training')
    meta = registry.set_active(model_id, meta.task)
    if meta:
        messages.success(request, f'Active model set to: {meta.name}')
    else:
        messages.error(request, 'Could not activate model.')
    return redirect(f'/ml/training/?task={meta.task if meta else "fake_news"}')


@require_POST
def model_delete(request, model_id):
    from .services.model_registry import ModelRegistry

    registry = ModelRegistry()
    if registry.delete(model_id):
        messages.success(request, 'Model deleted.')
    else:
        messages.error(request, 'Model not found.')
    return redirect('ml_engine:model_training')
