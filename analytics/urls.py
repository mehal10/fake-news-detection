from django.urls import path
from . import views

app_name = 'analytics'

urlpatterns = [
    path('', views.reports, name='index'),
    path('export/csv/', views.export_csv, name='export_csv'),
    path('export/metrics/', views.export_metrics, name='export_metrics'),
    path('export/pdf/', views.export_pdf, name='export_pdf'),
]
