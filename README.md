# Weather Type Classification
This project aims to classify weather types based on various meteorological features using machine learning techniques. The model is built using a Random Forest Classifier and deployed as a Streamlit web application.

## Project Overview
The goal of this project is to predict the type of weather based on several features such as temperature, humidity, wind speed, precipitation, atmospheric pressure, UV index, visibility, cloud cover, season, and location. The project includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment.

## Dataset
The dataset contains meteorological features along with the target variable 'Weather Type'. It includes both continuous and categorical data. Ensure the dataset is placed in the correct directory as specified in the code.

## Installation
To get started, clone this repository and install the required dependencies:


git clone https://github.com/your-username/weather-classification.git
cd weather-classification
pip install -r requirements.txt
## Usage
## Data Preprocessing and EDA

Load the dataset and perform initial exploration using pandas and skimpy.

Handle missing values and check for duplicates.

Separate continuous and categorical columns for analysis.

Visualize the data using seaborn for both continuous and categorical columns.

## Model Training

Encode categorical variables using LabelEncoder and scale continuous variables using StandardScaler.

Split the dataset into training and testing sets.

Train a Random Forest Classifier.

Perform hyperparameter tuning using GridSearchCV.

## Model Evaluation

Evaluate the model using metrics such as Mean Squared Error (MSE) and R^2 Score.

Save the trained model, encoders, and scaler using joblib.

## Web Application

Create a Streamlit web application for weather type prediction.

Preprocess input data using the saved encoders and scaler.

Make predictions using the trained model and display results.

To run the Streamlit app:


streamlit run app.py

## Model Training
### Training a Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

### Hyperparameter Tuning



from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
Model Evaluation
### Evaluate the Model


from sklearn.metrics import mean_squared_error, r2_score
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
## Web Application
### Streamlit App
Create a Streamlit app (app.py) for user interaction and prediction. Here’s a simplified version:


import streamlit as st
import joblib
import pandas as pd

# Load the model, encoders, and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Preprocess input data
def preprocess_input(...):  # Full function in project code
    ...

# Streamlit app
st.title('Weather Prediction App')
# User inputs
...
if st.button('Predict Weather Type'):
    preprocessed_data = preprocess_input(...)
    prediction = model.predict(preprocessed_data)
    st.write(f'Predicted Weather Type: {prediction[0]}')
## Future Work
Improve the model by experimenting with different algorithms and feature engineering techniques.
Enhance the web application with additional functionalities such as visualizing prediction probabilities and historical weather data.
Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
