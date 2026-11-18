from django.urls import path
from . import views

app_name = 'ml_engine'

urlpatterns = [
    path('datasets/', views.dataset_list, name='dataset_list'),
    path('datasets/upload/', views.dataset_upload, name='dataset_upload'),
    path('datasets/<int:pk>/', views.dataset_detail, name='dataset_detail'),
    path('datasets/<int:pk>/preprocess/', views.dataset_preprocess, name='dataset_preprocess'),
    path('datasets/<int:pk>/delete/', views.dataset_delete, name='dataset_delete'),
    path('datasets/<int:pk>/remap/', views.dataset_remap, name='dataset_remap'),
    path('preprocess/', views.preprocess_demo, name='preprocess_demo'),
    path('preprocess/api/', views.preprocess_api, name='preprocess_api'),
    path('training/', views.model_training, name='model_training'),
    path('training/start/', views.model_train_start, name='model_train_start'),
    path('models/<str:model_id>/', views.model_detail, name='model_detail'),
    path('models/<str:model_id>/activate/', views.model_set_active, name='model_set_active'),
    path('models/<str:model_id>/delete/', views.model_delete, name='model_delete'),
]
