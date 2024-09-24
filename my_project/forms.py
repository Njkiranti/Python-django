from django import forms
from .models import Staff,DeviceLocation
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class StaffForm(forms.ModelForm):
    class Meta:
        model = Staff
        fields = ['username', 'password', 'firstname', 'middlename', 'lastname', 'mobileno', 'gender', 'email', 'position', 'hash_value', 'isactive']
def __init__(self, *args, **kwargs):
        super(StaffForm, self).__init__(*args, **kwargs)

class UploadFileForm(forms.Form):
    file = forms.FileField()

class UserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username',) 

class DeviceLocationForm(forms.ModelForm):
    class Meta:
        model = DeviceLocation
        fields = ['device', 'latitude', 'longitude']

class LoginForm(forms.Form):
    username = forms.CharField(label="Username")
    password = forms.CharField(label="Password", widget=forms.PasswordInput)

