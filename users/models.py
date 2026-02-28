from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models
from django.core.validators import RegexValidator, EmailValidator

class UserRegistrationModel(models.Model):
    # Validators
    name_validator = RegexValidator(regex=r'^[a-zA-Z]+$', message='Name must contain only letters')
    loginid_validator = RegexValidator(regex=r'^[a-zA-Z]+$', message='Login ID must contain only letters')
    password_validator = RegexValidator(
        regex=r'(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}',
        message='Password must contain at least one number, one uppercase, one lowercase letter, and at least 8 characters'
    )
    mobile_validator = RegexValidator(regex=r'^[56789][0-9]{9}$', message='Mobile must start with 5, 6, 7, 8, or 9 and be 10 digits long')
    email_validator = EmailValidator(message='Enter a valid email address')

    # Fields with validation
    name = models.CharField(max_length=100, validators=[name_validator])
    loginid = models.CharField(max_length=100, validators=[loginid_validator])
    password = models.CharField(max_length=100, validators=[password_validator])
    mobile = models.CharField(max_length=10, validators=[mobile_validator])
    email = models.EmailField(max_length=100, validators=[email_validator])
    locality = models.CharField(max_length=100)
    address = models.TextField(max_length=250)
    status = models.CharField(max_length=100, default='waiting')

    def __str__(self):
        return self.name
    class meta:
        db='userRegistrationModel'


