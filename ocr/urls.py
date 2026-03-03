from django.urls import path
from . import views, auth


urlpatterns = [
        path('api/cheque', views.cheque_upload, name='cheque'),
        path('api/account', views.account, name='account'),
        path('api/foreign/', views.foreign, name='foreign'),
        path('api/foreign-confirm/', views.foreign_confirm_view, name='foreign-confirm'),

        path('api/login', auth.login_user, name="login"),
        path('api/logout', auth.logout_user, name="logout"),
        path('api/token', auth.prof, name="token"),
        path('api/confirm', views.confirm_view, name="confirm"),
]
