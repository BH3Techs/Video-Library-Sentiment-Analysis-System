from django.urls import path
from django.conf.urls import url

from .views import (PostListView, PostDetailView, PostCreateView, PostUpdateView)
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('login/', views.login, name="login"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('register/', views.register, name="register"),
    path('forgot_password/', views.register, name="forgot_password"),
    path('buttons/', views.buttons, name="buttons"),
    path('report/', views.report, name="report"),
    path('test/', views.test, name="test"),
    path('charts/', views.charts, name="charts"),
    path('comment/', PostListView.as_view(), name='comment'),
    path('post/new/', PostCreateView.as_view(), name='post-create'),
    path('post/<int:pk>/', PostListView.as_view(), name='post-detail'),
    path('post/<int:pk>/update/', PostUpdateView.as_view(), name='post-update'),
    path('dashboard/predictionAnnual/', views.predictionAnnual, name='predictionAnnual'),
    path('dashboard/predictionRange/', views.predictionRange, name='predictionRange'),
    path('dashboard/predictionWeek/', views.predictionWeek, name='predictionWeek'),
    path('dashboard/predictionToday', views.predictionToday, name='predictionToday'),
    path('dashboard/predictionMonth/', views.predictionMonth, name='predictionMonth'),
    path('dashboard/population-chart/', views.population_chart, name='population-chart'),
    



    
    
]