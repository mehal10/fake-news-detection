from django.urls import path
from . import views

app_name = 'combined_analysis'

urlpatterns = [
    path('', views.combined, name='index'),
]
