
from django.urls import path, include
from . import views

urlpatterns = [
    path('',views.emp_form, name='my_project_insert'),
    path('<int:id>/',views.emp_form,name='my_project_update'),
    path('delete/<int:id>/',views.emp_delete,name='my_project_delete'),
    path('list/', views.emp_list, name='list'),
    path('user_behaviour/', views.analyze_user_behavior_view, name='user_behaviour'),
    path('piracy_detection/', views.piracy_detection_view, name='piracy_detection'),
    # path('login/', views.login_view, name='login'),
    path('user_list/', views.user_list, name='user_list'),
    path('user_form/', views.user_create, name='user_form'),
    path('user_update/', views.user_update, name='user_update'),
    path('user_delete/<int:pk>/', views.user_delete, name='user_delete'),
    path('login/', views.user_login, name='user_login'),
    path('location/', views.device_location_view, name='device_location_view'),
    path('location_sucess/', views.device_location_view, name='location_sucess'),
    path('upload/', views.upload_file, name='upload_file'),
    path('hashing/', views.file_upload_view, name='hashing'),
    path('chart/', views.chart_view, name='chart'),
    path('bar_chart/', views.barchart_view, name='barchart'),
    path('pie_chart/', views.piechar_view, name='piechart'),
    path('line_graph/', views.line_graph_view, name='linegraph'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('send_mail/', views.send_email, name='send_mail'),
     path('piracy_data/', views.piracy_data, name='piracy_data'),
    path('user_behaviour_data/', views.user_behaviour_data, name='user_behaviour_data'),

    






 ]
