from django.shortcuts import render,redirect
from .hash_funtions import generate_hash,compare_hash_with_database
import pandas as pd
from django.http import HttpResponseRedirect
from .forms import UploadFileForm,UserForm,DeviceLocationForm,StaffForm,LoginForm
from .models import Data,Staff
import csv
import json
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objs as go
from django.core.mail import EmailMessage
import random
import os
from django.conf import settings
# Create your views here.

def emp_list(request):
    context = {'myapp_list': Staff.objects.all()}
    return render(request,"my_project/list.html",context)

def emp_form(request, id=0):
    if request.method == "GET":
      if id == 0:         
         form = StaffForm()
      else:
         employee = Staff.objects.get(pk=id)
         form = StaffForm(instance=employee)
      return render(request,"my_project/form.html",{'forms':form})
    
    elif request.method == "POST":
        if id == 0:
            form = StaffForm(request.POST)
        else:
            employee = Staff.objects.get(pk=id)
            form = StaffForm(request.POST, instance=employee)
        
        if form.is_valid():
            form.save()
            return redirect('my_project/list.html')  # Redirect to the list page after successful form submission
        else:
            # If form is not valid, return the form with error messages
            return render(request, "my_project/form.html", {'form': form})
       
def emp_delete(request,id=0):
         user = StaffForm.objects.get(pk=id)
         user.delete()
         return render(request,"my_project/list.html")

def file_upload_view(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']
        
        # Read the contents of the uploaded file
        file_contents = uploaded_file.read().decode('utf-8')
        
        # Calculate the hash of the file
        file_hash = generate_hash(file_contents)
        
        # Compare the hash with the database
        match = compare_hash_with_database(file_hash)
        
        if match:
            # If there's a match, perform appropriate action
            return render(request, 'my_project/hashing.html', {'result': "File hash matched with the database."})
        else:
            # If no match found, perform another action
            return render(request, 'my_project/hashing.html', {'result': "File hash did not matched with the database."})
    
    # If not a POST request or no file uploaded, render the form
    return render(request, 'my_project/hashing.html')

def analyze_user_behavior_view(request):
    predicted_labels = None
    if request.method == 'POST':
        feature1 = float(request.POST.get('feature1'))
        feature2 = float(request.POST.get('feature2'))
        feature3 = float(request.POST.get('feature3'))

        # Prepare input features as a DataFrame
        input_features = pd.DataFrame({'feature1': [feature1], 'feature2': [feature2], 'feature3': [feature3]})

        # Predict piracy label using the trained model
        # predicted_labels = classifier.predict(input_features)
        csv_file_path = os.path.join(settings.BASE_DIR, 'my_project','user_behavior_data.csv')

        user_behavior_data = pd.read_csv(csv_file_path)

        # Convert user behavior data to feature matrix X and target vector y
        X = user_behavior_data[['feature1', 'feature2', 'feature3']]
        y = user_behavior_data['piracy_label']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Predict labels for test data
        y_pred = classifier.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Example usage of trained model
        # You can use this model to analyze new user behavior data
        new_user_behavior_data = input_features
        predicted_labels = classifier.predict(new_user_behavior_data)
        response_data = {
                'accuracy': accuracy,
                'predicted_labels': predicted_labels
            } 
    return render(request, 'my_project/user_behaviour.html', {'result': response_data})

def piracy_detection_view(request):
    try:
        is_pirate = None
        if request.method == 'POST':
            # Extract parameters from the request
            username = request.POST.get('username')
            password = request.POST.get('password')
            ip_address = request.POST.get('ip_address')
            device_info = request.POST.get('device_info')

            # Call piracy_detection function
            if not all([username, password, ip_address, device_info]):
             raise ValueError("Missing input parameters")
            csv_file_path = os.path.join(settings.BASE_DIR, 'my_project','piracy_data.csv')
            data = pd.read_csv(csv_file_path)

            X = data.drop('is_pirate', axis=1)
            y = data['is_pirate']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            categorical_features = ['username', 'device_info']
            categorical_indices = [X.columns.get_loc(col) for col in categorical_features]
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            preprocessor = ColumnTransformer(
              transformers=[
                ('cat', categorical_transformer, categorical_indices)
            ])

            model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

            model.fit(X_train, y_train)

            features = [[username, password, ip_address, device_info]]
            is_pirate = model.predict(features)[0]

            # Calculate accuracy of the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            response_data = {
                'accuracy': accuracy,
                'piracy_status': is_pirate
            }
            # Return the result as JSON response
            return render(request, 'my_project/piracy_detection.html', {'result': response_data})
            # return JsonResponse({'accuracy': result['accuracy'],'is_pirate': result['is_pirate']})
        else:
        # If the request method is not POST, return the template
         return render(request, 'my_project/piracy_detection.html')

    except Exception as e:
        # Log the exception or print it for debugging
        print(e)
        # Return an error response
        return render(request, 'my_project/error.html', {'error_message': str(e)})

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            return HttpResponseRedirect('/success/')
    else:
        form = UploadFileForm()
    return render(request, 'my_project/upload.html', {'form': form})

def handle_uploaded_file(f):
    reader = csv.reader(f)
    for row in reader:
        _, created = Data.objects.get_or_create(name=row[0], value=row[1])

def chart_view(request):
    
    pirated_data = [
        {'name': f'Pirated {i}', 'value': random.randint(1, 100)} for i in range(1, 6)
    ]
    unpirated_data = [
        {'name': f'Unpirated {i}', 'value': random.randint(1, 100)} for i in range(1, 6)
    ]
    
    # Combine pirated and unpirated data
    combined_data = pirated_data + unpirated_data
    
    # Convert combined data to JSON
    data_json = json.dumps(combined_data)
    return render(request, 'my_project/chart.html', {'data_json': data_json})

def barchart_view(request):
    pirated_data = [
        {'name': f'Pirated {i}', 'value': random.randint(1, 100)} for i in range(1, 6)
    ]
    unpirated_data = [
        {'name': f'Unpirated {i}', 'value': random.randint(1, 100)} for i in range(1, 6)
    ]
    # data = list(Data.objects.values('name', 'value'))
    combined_data = pirated_data + unpirated_data
    data_json = json.dumps(combined_data)
    return render(request, 'my_project/bar_chart.html', {'data_json': data_json})
def piechar_view(request):
    pirated_data = [
        {'name': f'Pirated {i}', 'value': random.randint(1, 100)} for i in range(1, 6)
    ]
    unpirated_data = [
        {'name': f'Unpirated {i}', 'value': random.randint(1, 100)} for i in range(1, 6)
    ]
    # data = list(Data.objects.values('name', 'value'))
    combined_data = pirated_data + unpirated_data
    data_json = json.dumps(combined_data)
    return render(request, 'my_project/pie_chart.html', {'data_json': data_json})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('my_project/hashing.html')  # Redirect to home page after successful login
        else:
            send_piracy_alert_email(request)
            return render(request, 'my_project/login.html', {'error_message': 'Invalid username or password'})
    return render(request, 'my_project/login.html')


def user_list(request):
    users = User.objects.all()
    if users:
        return render(request, 'my_project/user_list.html', {'users': users})
    else:
        # If no users found, render the template without passing u20sddsers data
        return render(request, 'my_project/user_list.html')

def user_create(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            # Extract username and password from the form data
            username = form.cleaned_data['username']
            password1 = form.cleaned_data['password1']
            password2 = form.cleaned_data['password2']
            
            if password1 != password2:
                messages.error(request, "Passwords do not match.")
                return redirect('my_project/user_form')
            
            # Hash the password before saving it to the database
            hashed_password = make_password(password1)
            
            new_user = form.save(commit=False)  # Create a new user object but don't save it to the database yet
            new_user.set_password(password1)  # Set the hashed password
            new_user.save()
            messages.success(request, 'User {} has been created successfully.'.format(username))
            return redirect('user_list')  # Redirect to user_list view after user creation
    else:
        form = UserForm()
    return render(request, 'my_project/user_form.html', {'form': form})

def user_update(request, pk):
    user = UserForm.objects.get(pk=pk)
    if request.method == 'POST':
        form = UserForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return redirect('my_project/user_list')
    else:
        form = UserForm(instance=user)
    return render(request, 'my_project/user_form.html', {'form': form})

def user_delete(request, pk):
    try:
        user = UserForm.objects.get(pk=pk)
    except UserForm.DoesNotExist:
        # Handle the case where the user does not exist
        # You can redirect to an error page or display a message to the user
        return render(request, 'my_project/user_form.html')
    
    if request.method == 'POST':
        # If the request method is POST, it means the user has confirmed deletion
        user.delete()
        # Redirect to the user list page after deletion
        return redirect('my_project/user_list.html')

    # If the request method is GET, render the confirmation page
    return render(request, 'my_project/user_delete.html', {'user': user})

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
             # Check if the user exists
          
            user  = authenticate(request, username=username, password=password)
            print(user )
            if user  is not None:

                login(request, user )
                messages.success(request, 'Successfully logged in.')
                return render(request, 'my_project/dashboard.html')
                # return redirect(request,"user_list")  # Redirect to the desired page after successful login
            else:
                send_piracy_alert_email(request)
                messages.error(request, 'Invalid username or password.')
        else:
            print(form.errors)
    else:
        form = LoginForm() 
    return render(request, 'my_project/login.html', {'form': form})

def send_piracy_alert_email(request):
    """Send a piracy alert email to the system administrator."""
    username = request.GET.get('username')  # Assuming username is passed as a query parameter
    if not username:
        return HttpResponse("Username not provided.", status=400)

    # Replace these with your actual Gmail credentials
    sender_email = "njkiranti0@gmail.com"  # Replace with your Gmail email address
    receiver_email = "nirajrai557@gmail.com"  # Replace with the recipient's email address
    password = "gzdj iwfo wyfp ktji"  # Replace with your Gmail password
    subject = "Piracy Alert: Unauthorized Software Usage Detected"
    body = f"Username: {username}\nUnauthorized software usage has been detected."

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        return HttpResponse("Piracy alert email sent to the system administrator.", status=200)
    except Exception as e:
        return HttpResponse(f"Error sending email: {e}", status=500)

def device_location_view(request):
    if request.method == 'POST':
        form = DeviceLocationForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'my_project/location_success.html')
    else:
        form = DeviceLocationForm()
    return render(request, 'my_project/location.html', {'form': form})

def dashboard(request):
    return render(request, 'my_project/dashboard.html')


# def piracy_detection(username, password, ip_address, device_info):
#     try:
#         if not all([username, password, ip_address, device_info]):
#             raise ValueError("Missing input parameters")
#         csv_file_path = os.path.join(settings.BASE_DIR, 'my_project','piracy_data.csv')
#         data = pd.read_csv(csv_file_path)

#         X = data.drop('is_pirate', axis=1)
#         y = data['is_pirate']
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         categorical_features = ['username', 'device_info']
#         categorical_indices = [X.columns.get_loc(col) for col in categorical_features]
#         categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#         preprocessor = ColumnTransformer(
#             transformers=[
#                 ('cat', categorical_transformer, categorical_indices)
#             ])

#         model = Pipeline(steps=[('preprocessor', preprocessor),
#                                 ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         features = [[username, password, ip_address, device_info]]
#         is_pirate = model.predict(features)[0]
#         response_data = {
#                 'accuracy': accuracy,
#                 'piracy_status': is_pirate
#             }
#         return response_data
#     except Exception as e:
#         # Log the exception or print it for debugging
#         print(e)
#         # Return an error response
#         return {'error': 'An error occurred'}
    
def analyze_user_behavior(data):
    try:
        # Read user behavior data from CSV file
        csv_file_path = os.path.join(settings.BASE_DIR, 'my_project','user_behavior_data.csv')

        user_behavior_data = pd.read_csv(csv_file_path)

        # Convert user behavior data to feature matrix X and target vector y
        X = user_behavior_data[['feature1', 'feature2', 'feature3']]
        y = user_behavior_data['piracy_label']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Predict labels for test data
        y_pred = classifier.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Example usage of trained model
        # You can use this model to analyze new user behavior data
        new_user_behavior_data = data
        predicted_labels = classifier.predict(new_user_behavior_data)
        response_data = {
                'accuracy': accuracy,
                'predicted_labels': predicted_labels
            }
        return response_data
    except Exception as e:
        # Log the exception or print it for debugging
        print(e)
        # Return an error response
        return {'error': 'An error occurred'}

def line_graph_view(request):
    # Sample data (replace this with your actual data)
    x_values = [1, 2, 3, 4, 5]
    y_values = [2, 3, 5, 4, 6]

    # Create a Plotly trace
    trace = go.Scatter(x=x_values, y=y_values, mode='lines')

    # Create a Plotly layout
    layout = go.Layout(title='Line Graph', xaxis=dict(title='X Axis'), yaxis=dict(title='Y Axis'))

    # Create a Plotly figure
    fig = go.Figure(data=[trace], layout=layout)

    # Convert the Plotly figure to JSON
    graph_json = fig.to_json()

    # Pass the JSON data to the template
    return render(request, 'my_project/line_graph.html', {'graph_json': graph_json})



def piracy_data(request):
    # Read CSV file
    csv_file_path = os.path.join(settings.BASE_DIR, 'my_project','piracy_data.csv')
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Pass data to the template
    return render(request, 'my_project/piracy_data.html', {'data': data})

def user_behaviour_data(request):
    # Read CSV file
    csv_file_path = os.path.join(settings.BASE_DIR,'my_project', 'user_behavior_data.csv')
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Pass data to the template
    return render(request, 'my_project/user_behaviour_data.html', {'data': data})

def send_email(request):
    if request.method == 'POST':
        sender_email = request.POST.get('sender_email')
        receiver_email = request.POST.get('receiver_email')
        password = request.POST.get('password')
        message = request.POST.get('message')

        if not (sender_email and receiver_email and password and message):
            return HttpResponse("Please fill out all fields.", status=400)

        try:
            email = EmailMessage(
                subject="Piracy Alert: Unauthorized Software Usage Detected",
                body=message,
                from_email=sender_email,
                to=[receiver_email],
            )
            email.send(fail_silently=False)
            return render(request, 'my_project/send_mail.html',{'result':"Piracy alert email sent to the system administrator."})
            
        except Exception as e:
            return render(request, 'my_project/send_mail.html',{'result':"Error sending email."})
            
    else:
        return render(request, 'my_project/send_mail.html')
        