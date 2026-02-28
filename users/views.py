import warnings
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from matplotlib import pyplot as plt
import pandas as pd
import os

from .models import UserRegistrationModel

def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST.get('mobile', ''),
            email=request.POST['email'],
            locality=request.POST.get('locality', ''),
            address=request.POST.get('address', ''),
            # city=request.POST.get('city', ''),
            # state=request.POST.get('state', ''),
            status='waiting'
        )
        user.save()
        messages.success(request, "Registration successful! Please wait for admin approval.")
        return redirect('UserLogin')
    return render(request, 'UserRegistration.html')


# def UserLoginCheck(request):
#     if request.method == "POST":
#         loginid = request.POST.get('loginid')
#         pswd = request.POST.get('password')  # Corrected from 'pswd'

#         try:
#             check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
#             if check.status == "activated":
#                 request.session['id'] = check.id
#                 request.session['loggeduser'] = check.name
#                 request.session['loginid'] = loginid
#                 request.session['email'] = check.email
#                 return redirect('UserHome')
#             else:
#                 messages.warning(request, 'Your account is not yet activated. Please wait for admin approval.')
#                 return redirect('UserLogin')
#         except UserRegistrationModel.DoesNotExist:
#             messages.error(request, 'Invalid login ID or password.')
#             return redirect('UserLogin')

#     return render(request, 'login.html')  

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserRegistrationModel

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('password')

        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "active":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return redirect('UserHome')
            else:
                messages.warning(request, 'Your account is not yet activated. Please wait for admin approval.')
                return redirect('UserLogin')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid login ID or password.')
            return redirect('UserLogin')

    return render(request, 'login.html')



def UserHome(request):
    if not request.session.get('id'):
        return redirect('UserLogin')
    return render(request, 'users/UserHome.html')


def base(request):
    return render(request, 'base.html')


def index(request):
    return render(request, "index.html")


def logout_view(request):
    request.session.flush()
    return redirect('index')


def viewdataset(request):
    dataset_path = os.path.join(settings.MEDIA_ROOT, 'Hypertension-risk-model-main.csv')
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path).head(4000)
        table_html = df.to_html(classes='table table-striped table-bordered', index=False, border=0)
        return render(request, 'users/viewdataset.html', {'table': table_html})
    else:
        messages.error(request, "Dataset not found.")
        return redirect('UserHome')

    


################################################  ML CODE  ############################################################ 


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI from being used
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib  # Import joblib to save the model

warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

selected_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

def training(request):
    if request.method == 'POST':
        data = pd.read_csv('media/Hypertension-risk-model-main.csv')  # Use forward slash for cross-platform
        data.replace('NA', np.nan, inplace=True)

        X = data[selected_features]
        y = data['Risk']

        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC(),
            'XGBoost': XGBClassifier(eval_metric='logloss')
        }

        model_results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            # Plot confusion matrix and encode it
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()  # Prevent overlapping plots

            cm_uri = base64.b64encode(image_png).decode('utf-8')

            model_results[name] = {
                'accuracy': f"{acc:.4f}",
                'confusion_matrix': cm_uri,
                'classification_report': report
            }

            # Save each trained model
            joblib.dump(model, f'media/{name}_model.pkl')  # Save model with its name

        # Save the scaler and imputer for future use in prediction
        joblib.dump(scaler, 'media/scaler.pkl')
        joblib.dump(imputer, 'media/imputer.pkl')

        context = {
            'status': 'Training Completed',
            'model_results': model_results
        }

        return render(request, 'users/training.html', context)

    return render(request, 'users/training.html')


import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render

# Load the pre-trained model and scaler
MODEL_PATH = 'media/Random Forest_model.pkl'  # Update to the desired model file
SCALER_PATH = 'media/scaler.pkl'
IMPUTER_PATH = 'media/imputer.pkl'

def predict(request):
    if request.method == 'POST':
        try:
            # Load the pre-trained model, scaler, and imputer
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            imputer = joblib.load(IMPUTER_PATH)

            # Get input data from the form
            age = float(request.POST.get('age'))
            cigsPerDay = float(request.POST.get('cigsPerDay'))
            totChol = float(request.POST.get('totChol'))
            sysBP = float(request.POST.get('sysBP'))
            diaBP = float(request.POST.get('diaBP'))
            BMI = float(request.POST.get('BMI'))
            heartRate = float(request.POST.get('heartRate'))
            glucose = float(request.POST.get('glucose'))

            input_data = pd.DataFrame([[age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose]],
                                      columns=['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])

            # Impute missing values
            input_imputed = imputer.transform(input_data)

            # Scale the data
            input_scaled = scaler.transform(input_imputed)

            # Predict the result
            prediction = model.predict(input_scaled)[0]
            risk_label = 'High Risk' if prediction == 1 else 'Low Risk'

            context = {
                'prediction': risk_label,
                'input': input_data.iloc[0].to_dict()
            }

            return render(request, 'users/prediction.html', context)

        except Exception as e:
            return render(request, 'users/prediction.html', {'error': str(e)})

    return render(request, 'users/prediction.html')
