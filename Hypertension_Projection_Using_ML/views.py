from django.shortcuts import render


def index(request):
    return render(request,'index.html')
def userRegistration(request):
    return render(request,'userRegistration.html')
def userLogin(request):
    return render(request,'login.html')
def adminLogin(request):
    return render(request,'adminlogin.html')