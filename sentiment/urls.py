from django.urls import path
from . import views

app_name = 'sentiment'

urlpatterns = [
    path('', views.analyze, name='index'),
]
