from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views


urlpatterns = [
        path('', views.home, name='home'),
        path('api/token/', TokenObtainPairView.as_view(), name='token-obtain'),
        path('api/token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),
        path('api/foreign/', views.foreign, name='foreign'),
        path('api/foreign-confirm/', views.foreign_confirm_view, name='foreign-confirm'),

        path('login', views.login_user, name="login"),
        path('logout', views.logout_user, name="logout"),
        path('confirm', views.confirm_view, name="confirm"),
]
