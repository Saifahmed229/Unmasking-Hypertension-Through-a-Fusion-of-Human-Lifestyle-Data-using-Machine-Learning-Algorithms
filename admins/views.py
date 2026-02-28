from django.shortcuts import redirect, render
from django.contrib import messages
from users.models import UserRegistrationModel

# Admin login check function
def adminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('password')
        print("User ID is = ", usrid)
        print("Password = ", pswd)

        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/adminHome.html')
        else:
            messages.error(request, 'Please check your login details.')
            return render(request, 'adminlogin.html')
    
    # ✅ Handle GET request
    return render(request, 'adminlogin.html')


# Admin home page
def adminHome(request):
    return render(request, 'admins/adminHome.html')

# View all registered users
def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def activateUser(request, id):
    print(f"Activating user with ID = {id}")
    UserRegistrationModel.objects.filter(id=id).update(status='active')
    return redirect('RegisterUsersView')  # Ensure this matches your URL name

def deactivateUser(request, id):
    print(f"Deactivating user with ID = {id}")
    UserRegistrationModel.objects.filter(id=id).update(status='deactivated')
    return redirect('RegisterUsersView')

def deleteUser(request, id):
    print(f"Deleting user with ID = {id}")
    UserRegistrationModel.objects.filter(id=id).delete()
    return redirect('RegisterUsersView')