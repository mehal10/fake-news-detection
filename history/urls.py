from django.urls import path
from . import views

app_name = 'history'

urlpatterns = [
    path('', views.history_list, name='index'),
    path('<int:pk>/', views.history_detail, name='detail'),
    path('<int:pk>/delete/', views.history_delete, name='delete'),
    path('delete-all/', views.history_delete_all, name='delete_all'),
    path('<int:pk>/export/', views.history_export, name='export'),
    path('export/', views.history_export_list, name='export_list'),
]
