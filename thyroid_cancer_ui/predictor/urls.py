from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_recurrence, name='predict_recurrence'),
]

