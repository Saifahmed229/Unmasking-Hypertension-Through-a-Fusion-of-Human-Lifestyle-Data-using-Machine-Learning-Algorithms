"""
URL configuration for Hypertension_Projection_Using_ML project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

# from django.contrib import admin
# from django.urls import path
# from . import views as mv
# from users import views as uv
# from admins import views as av
# from django.conf import settings
# from django.conf.urls.static import static
# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('',mv.index,name='index'),
#     path('userRegistration',mv.userRegistration,name='userRegistration'),
#     path('userLogin',mv.userLogin,name='userLogin'),
#     path('adminLogin',mv.adminLogin,name='adminLogin'),

#     #userUrlsssssssss
#     path('',uv.base,name='base'),
#     path('userHome',uv.userHome,name='userHome'),
#     path('userRegister',uv.userRegister,name='userRegister'),
#     path('userLoginCheck',uv.userLoginCheck,name='userLoginCheck'),

#     path('userViewdataset',uv.view_dataset,name='userViewDataset'),
#     # path('train_model_view',uv.train_model_view,name='train_model_view'),
#     # path('prediction',uv.predict_image_view,name='predict_image_view'),

#     #adminurls
#     path('adminHome',av.adminHome,name='adminHome'),
#     path('adminLoginCheck', mv.adminLogin, name='adminLoginCheck'),
#     path('RegisterUsersView',av.RegisterUsersView,name='RegisterUsersView'),
#     path('activateUser',av.activateUser,name='activateUser'),
#     path('deleteUser',av.deleteUser,name='deleteUser'),
  

   
# ]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)  



# from django.contrib import admin
# from django.urls import path
# from django.conf import settings
# from django.conf.urls.static import static

# from . import views as mv
# from admins import views as admins
# from users import views as usr  # unified import

# urlpatterns = [
#     # Django Admin
#     path('admin/', admin.site.urls),

#     # Landing & Authentication
#     path('', mv.index, name='index'),
#     path('index/', mv.index, name='index'),
#     path('Adminlogin/', mv.adminLogin, name='AdminLogin'),
#     path('UserLogin/', mv.userLogin, name='UserLogin'),
#     path('UserLoginCheck/', usr.UserLoginCheck, name='UserLoginCheck'),
#     path('UserRegisterForm/', usr.UserRegisterActions, name='UserRegisterForm'),

#     # Admin Panel
#     path('Adminlogin/AdminLogincheck/', admins.adminLoginCheck, name='AdminLoginCheck'),
#     path('AdminHome/', admins.adminHome, name='AdminHome'),
#     path('userDetails/', admins.RegisterUsersView, name='RegisterUsersView'),
#     path('ActivUsers/', admins.activateUser, name='activate_users'),
#     path('DeleteUsers/', admins.deleteUser, name='delete_users'),

#     # User Panel
#     path('UserHome/', usr.UserHome, name='UserHome'),
#     path('base/',usr.base,name='base'),
#     path('viewdataset/', usr.view_dataset, name='viewdataset'),  # Added View Dataset
#     # path('train/', usr.train_view, name='train'),                # Added Training
#     # path('predict/', usr.predict_view, name='predict'),          # Added Prediction
#     path('logout/', usr.logout_view, name='logout'),             # Logout
# ]


from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views as mv
from admins import views as admins
from users import views as usr

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),

    # General
    path('', mv.index, name='index'),
    path('index/', mv.index, name='index'),
    path('UserLogin/', mv.userLogin, name='UserLogin'),
    path('Adminlogin/', mv.adminLogin, name='AdminLogin'),
    path('UserRegister/',mv.userRegistration, name='UserRegister'),


    path('base/', usr.base, name='base'),
    path('logout/', usr.logout_view, name='logout'),
    path('base/', usr.base, name='base'),
    path('logout/', usr.logout_view, name='logout'),
    path('UserLoginCheck/', usr.UserLoginCheck, name='UserLoginCheck'),
    path('UserRegisterForm/', usr.UserRegisterActions, name='UserRegisterForm'),
    path('UserHome/', usr.UserHome, name='UserHome'),
    path('viewdataset/', usr.viewdataset, name='viewdataset'), 


    path('AdminLogincheck/', admins.adminLoginCheck, name='AdminLoginCheck'),
    path('AdminHome/', admins.adminHome, name='AdminHome'),
    path('userDetails/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('activateUser/<int:id>/', admins.activateUser, name='activate_users'),
    path('deactivateUser/<int:id>/', admins.deactivateUser, name='deactivate_users'),
    path('deleteUser/<int:id>/', admins.deleteUser, name='delete_users'), 
    path('training/',usr.training,name='training'),
    path('prediction/',usr.predict,name='prediction'),

]

# Static media support
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

