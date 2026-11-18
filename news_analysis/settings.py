"""
Django settings for news_analysis project.
Fake News Detection and News Sentiment Analysis System
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Security — override via environment variables in production
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-news-analysis-dev-key-change-in-production-2026',
)

# Default True for local development. Set DJANGO_DEBUG=0 in production.
DEBUG = os.environ.get('DJANGO_DEBUG', '1') not in ('0', 'false', 'False')

ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get('DJANGO_ALLOWED_HOSTS', '127.0.0.1,localhost,testserver').split(',')
    if h.strip()
]
if DEBUG:
    for h in ('127.0.0.1', 'localhost', 'testserver'):
        if h not in ALLOWED_HOSTS:
            ALLOWED_HOSTS.append(h)

# Production hardening (active when DEBUG is False)
if not DEBUG:
    SECURE_BROWSER_XSS_FILTER = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
    SESSION_COOKIE_SECURE = os.environ.get('DJANGO_SSL', '0') in ('1', 'true', 'True')
    CSRF_COOKIE_SECURE = SESSION_COOKIE_SECURE
    X_FRAME_OPTIONS = 'DENY'

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'core',
    'dashboard',
    'detector',
    'sentiment',
    'combined_analysis',
    'history',
    'analytics',
    'ml_engine',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Serve static files in production
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'news_analysis.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django.template.context_processors.media',
            ],
        },
    },
]

WSGI_APPLICATION = 'news_analysis.wsgi.application'

# Persist SQLite under the project tree (not /tmp)
# SQLite database file (override with DJANGO_DB_PATH)
_DB_PATH = Path(os.environ.get('DJANGO_DB_PATH', str(BASE_DIR / 'db.sqlite3')))
try:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Probe journal support; some networked volumes break SQLite commits
    import sqlite3 as _sqlite3
    _c = _sqlite3.connect(str(_DB_PATH))
    _c.execute('CREATE TABLE IF NOT EXISTS _probe (id INTEGER)')
    _c.commit()
    _c.execute('DROP TABLE _probe')
    _c.commit()
    _c.close()
except Exception:
    _DB_PATH = Path('/tmp/news_analysis.db')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(_DB_PATH),
        'OPTIONS': {'timeout': 30},
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'Asia/Kolkata'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# WhiteNoise for efficient static file serving on Render
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedStaticFilesStorage",
    },
}

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Upload limits — datasets only; models are trained server-side
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10 MB in memory
DATA_UPLOAD_MAX_MEMORY_SIZE = 12 * 1024 * 1024
# Hard cap for CSV dataset uploads (enforced in views as well)
MAX_DATASET_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

from django.contrib.messages import constants as messages

MESSAGE_TAGS = {
    messages.DEBUG: 'debug',
    messages.INFO: 'info',
    messages.SUCCESS: 'success',
    messages.WARNING: 'warning',
    messages.ERROR: 'error',
}

# Logging — avoid leaking paths in production responses; log server-side only
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO' if not DEBUG else 'DEBUG',
    },
}
