from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# 1. Collect Data
def piracy_detection(username, password, ip_address, device_info):
    data = pd.read_csv('piracy_data.csv')

    # 2. Preprocess Data
    X = data.drop('is_pirate', axis=1)
    y = data['is_pirate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    categorical_features = ['username', 'device_info']
    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_indices)
        ])

    # Define the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])

    # Train the model
    model.fit(X_train, y_train)

    # 4. Evaluate Model Performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 5. Integrate with Login System
    features = [[username, password, ip_address, device_info]]
    is_pirate = model.predict(features)[0]
    response_data = {
            'accuracy': accuracy,
            'piracy_status': is_pirate
        }
    return JsonResponse(response_data)
