"""
Model registry: tracks trained models, active versions, and file paths.
Model binaries live under MEDIA_ROOT/models/ — not in SQLite.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from django.conf import settings


def _models_root() -> Path:
    root = Path(settings.MEDIA_ROOT) / 'models'
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class ModelMeta:
    model_id: str
    name: str
    task: str  # 'fake_news' | 'sentiment'
    algorithm: str  # naive_bayes | logistic_regression | word2vec_lr
    feature_method: str  # tfidf | word2vec
    dataset_name: str
    dataset_id: Optional[int]
    metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ''
    is_active: bool = False
    train_size: int = 0
    test_size: int = 0
    training_time_sec: float = 0.0
    testing_time_sec: float = 0.0
    status: str = 'ready'  # ready | error
    message: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMeta':
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})

    @property
    def model_dir(self) -> Path:
        return _models_root() / self.model_id

    @property
    def classifier_path(self) -> Path:
        return self.model_dir / 'classifier.joblib'

    @property
    def vectorizer_path(self) -> Path:
        return self.model_dir / 'vectorizer.joblib'

    @property
    def embedding_path(self) -> Path:
        return self.model_dir / 'word2vec.model'

    @property
    def meta_path(self) -> Path:
        return self.model_dir / 'meta.json'


class ModelRegistry:
    """File-based registry of trained models."""

    INDEX_FILE = 'registry_index.json'

    def __init__(self):
        self.root = _models_root()
        self.index_path = self.root / self.INDEX_FILE

    def _load_index(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {'models': {}, 'active': {}}
        with open(self.index_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_index(self, index: Dict[str, Any]) -> None:
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

    def create_id(self) -> str:
        return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]

    def register(self, meta: ModelMeta) -> ModelMeta:
        meta.model_dir.mkdir(parents=True, exist_ok=True)
        with open(meta.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta.to_dict(), f, indent=2)

        index = self._load_index()
        index['models'][meta.model_id] = meta.to_dict()
        self._save_index(index)
        return meta

    def get(self, model_id: str) -> Optional[ModelMeta]:
        index = self._load_index()
        data = index['models'].get(model_id)
        if not data:
            # try loading from disk
            meta_path = self.root / model_id / 'meta.json'
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                return None
        return ModelMeta.from_dict(data)

    def list_models(self, task: Optional[str] = None) -> List[ModelMeta]:
        index = self._load_index()
        result = []
        for mid, data in index.get('models', {}).items():
            meta = ModelMeta.from_dict(data)
            if task and meta.task != task:
                continue
            result.append(meta)
        result.sort(key=lambda m: m.created_at, reverse=True)
        return result

    def set_active(self, model_id: str, task: str) -> Optional[ModelMeta]:
        meta = self.get(model_id)
        if not meta or meta.task != task:
            return None
        index = self._load_index()
        # deactivate previous
        prev_id = index.get('active', {}).get(task)
        if prev_id and prev_id in index['models']:
            index['models'][prev_id]['is_active'] = False
            prev_meta_path = self.root / prev_id / 'meta.json'
            if prev_meta_path.exists():
                with open(prev_meta_path, 'r', encoding='utf-8') as f:
                    prev_data = json.load(f)
                prev_data['is_active'] = False
                with open(prev_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(prev_data, f, indent=2)

        index.setdefault('active', {})[task] = model_id
        index['models'][model_id]['is_active'] = True
        meta.is_active = True
        with open(meta.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta.to_dict(), f, indent=2)
        self._save_index(index)
        return meta

    def get_active(self, task: str) -> Optional[ModelMeta]:
        index = self._load_index()
        mid = index.get('active', {}).get(task)
        if not mid:
            # fallback: first ready model for task
            models = self.list_models(task=task)
            for m in models:
                if m.status == 'ready':
                    return m
            return None
        return self.get(mid)

    def delete(self, model_id: str) -> bool:
        import shutil
        meta = self.get(model_id)
        if not meta:
            return False
        if meta.model_dir.exists():
            shutil.rmtree(meta.model_dir, ignore_errors=True)
        index = self._load_index()
        index['models'].pop(model_id, None)
        for task, mid in list(index.get('active', {}).items()):
            if mid == model_id:
                del index['active'][task]
        self._save_index(index)
        return True
