"""
URL configuration for news_analysis project.
Fake News Detection and News Sentiment Analysis System
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('dashboard/', include('dashboard.urls')),
    path('detector/', include('detector.urls')),
    path('sentiment/', include('sentiment.urls')),
    path('combined/', include('combined_analysis.urls')),
    path('history/', include('history.urls')),
    path('analytics/', include('analytics.urls')),
    path('ml/', include('ml_engine.urls')),
]

# Always serve media files (required on Render free tier – no separate media server)
# Static files are handled by WhiteNoise in production
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0] if settings.STATICFILES_DIRS else settings.STATIC_ROOT)
