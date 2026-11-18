
"""
Django settings for news_analysis project.
Fake News Detection and News Sentiment Analysis System
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# SECURITY CONFIGURATION
# ============================================================

SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-news-analysis-dev-key-change-in-production-2026',
)

DEBUG = os.environ.get('DJANGO_DEBUG', '0').lower() in (
    '1', 'true', 'yes'
)

ALLOWED_HOSTS = [
    host.strip()
    for host in os.environ.get(
        'DJANGO_ALLOWED_HOSTS',
        '127.0.0.1,localhost,testserver,fake-news-detection-vri2.onrender.com'
    ).split(',')
    if host.strip()
]

CSRF_TRUSTED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        'DJANGO_CSRF_TRUSTED_ORIGINS',
        'https://fake-news-detection-vri2.onrender.com'
    ).split(',')
    if origin.strip()
]

if DEBUG:
    if 'testserver' not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append('testserver')


# ============================================================
# PRODUCTION SECURITY
# ============================================================

if not DEBUG:
    SECURE_SSL_REDIRECT = os.environ.get(
        'DJANGO_SSL_REDIRECT', '1'
    ).lower() in ('1', 'true', 'yes')

    SECURE_PROXY_SSL_HEADER = (
        'HTTP_X_FORWARDED_PROTO',
        'https'
    )

    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True

    SECURE_CONTENT_TYPE_NOSNIFF = True
    X_FRAME_OPTIONS = 'DENY'

    SECURE_REFERRER_POLICY = 'strict-origin-when-cross-origin'

    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True


# ============================================================
# APPLICATIONS
# ============================================================

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Project applications
    'core',
    'dashboard',
    'detector',
    'sentiment',
    'combined_analysis',
    'history',
    'analytics',
    'ml_engine',
]


# ============================================================
# MIDDLEWARE
# ============================================================

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',

    'whitenoise.middleware.WhiteNoiseMiddleware',

    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]


# ============================================================
# URL CONFIGURATION
# ============================================================

ROOT_URLCONF = 'news_analysis.urls'

WSGI_APPLICATION = 'news_analysis.wsgi.application'


# ============================================================
# TEMPLATES
# ============================================================

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',

        'DIRS': [
            BASE_DIR / 'templates',
        ],

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


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

# SQLite database.
# For Render persistence, set DJANGO_DB_PATH to a file
# inside a mounted persistent disk.

DB_PATH = os.environ.get(
    'DJANGO_DB_PATH',
    str(BASE_DIR / 'db.sqlite3')
)

_DB_PATH = Path(DB_PATH)

# Create the parent directory if it does not exist.
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': str(_DB_PATH),
        'OPTIONS': {
            'timeout': 30,
        },
    }
}


# ============================================================
# PASSWORD VALIDATION
# ============================================================

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': (
            'django.contrib.auth.password_validation.'
            'UserAttributeSimilarityValidator'
        ),
    },
    {
        'NAME': (
            'django.contrib.auth.password_validation.'
            'MinimumLengthValidator'
        ),
    },
    {
        'NAME': (
            'django.contrib.auth.password_validation.'
            'CommonPasswordValidator'
        ),
    },
    {
        'NAME': (
            'django.contrib.auth.password_validation.'
            'NumericPasswordValidator'
        ),
    },
]


# ============================================================
# INTERNATIONALIZATION
# ============================================================

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'Asia/Kolkata'

USE_I18N = True

USE_TZ = True


# ============================================================
# STATIC FILE CONFIGURATION
# ============================================================

STATIC_URL = '/static/'

STATICFILES_DIRS = [
    BASE_DIR / 'static',
]

STATIC_ROOT = BASE_DIR / 'staticfiles'

STORAGES = {
    'default': {
        'BACKEND': 'django.core.files.storage.FileSystemStorage',
    },

    'staticfiles': {
        'BACKEND': (
            'whitenoise.storage.CompressedManifestStaticFilesStorage'
        ),
    },
}


# ============================================================
# MEDIA FILE CONFIGURATION
# ============================================================

MEDIA_URL = '/media/'

MEDIA_ROOT = BASE_DIR / 'media'


# ============================================================
# DEFAULT PRIMARY KEY
# ============================================================

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ============================================================
# FILE UPLOAD CONFIGURATION
# ============================================================

FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024

DATA_UPLOAD_MAX_MEMORY_SIZE = 12 * 1024 * 1024

MAX_DATASET_UPLOAD_BYTES = 25 * 1024 * 1024


# ============================================================
# DJANGO MESSAGES
# ============================================================

from django.contrib.messages import constants as messages

MESSAGE_TAGS = {
    messages.DEBUG: 'debug',
    messages.INFO: 'info',
    messages.SUCCESS: 'success',
    messages.WARNING: 'warning',
    messages.ERROR: 'error',
}


# ============================================================
# LOGGING
# ============================================================

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


# ============================================================
# SESSION CONFIGURATION
# ============================================================

SESSION_COOKIE_HTTPONLY = True

SESSION_COOKIE_SAMESITE = 'Lax'

CSRF_COOKIE_SAMESITE = 'Lax'
