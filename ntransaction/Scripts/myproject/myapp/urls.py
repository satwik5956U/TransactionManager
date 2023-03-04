from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('home',views.index,name="home"),
    path('signup',views.signup,name="signup"),
    path('signin',views.signin,name="signin"),
    path('signupdata',views.signupdata,name="signupdata"),
    path('signindata',views.signindata,name="signindata"),
    path('creditbutton',views.creditbutton,name="creditbutton"),
    path('creditpage',views.creditpage,name="creditpage"),
    path('ministate',views.ministate,name='ministate'),
    path('debitbutton',views.debitbutton,name="debitbutton"),
    path('debitpage',views.debitpage,name="debitpage"),
]
