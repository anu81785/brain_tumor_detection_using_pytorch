from django.urls import path
from . import views

urlpatterns = [
    path('', views.classify_tumor, name='upload'),
    # path('result/', views.result, name='result'),
]