import pandas as pd
import streamlit as st
import joblib

# Load the model, encoders, and scaler
model = joblib.load('random_forest_model.pkl')
one_hot_encoder = joblib.load('label_encoder_Cloud Cover.pkl')  # Load OneHotEncoder if saved separately
scaler = joblib.load('scaler.pkl')  # Load StandardScaler if saved separately

# Create a function to preprocess input data
def preprocess_input(temp, humidity, wind_speed, precip, pressure, uv_index, visibility, cloud_cover, season, location):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Temperature': [temp],
        'Humidity': [humidity],
        'Wind Speed': [wind_speed],
        'Precipitation (%)': [precip],
        'Atmospheric Pressure': [pressure],
        'UV Index': [uv_index],
        'Visibility (km)': [visibility],
        'Cloud Cover': [cloud_cover],
        'Season': [season],
        'Location': [location]
    })

    # Apply one-hot encoding
    input_data_encoded = pd.get_dummies(input_data, columns=['Cloud Cover', 'Season', 'Location'], drop_first=True)

    # Apply scaling
    input_data_scaled = scaler.transform(input_data_encoded)

    return input_data_scaled

# Define Streamlit app
st.title('Weather Prediction App')

# Input fields for the user
temp = st.number_input('Temperature')
humidity = st.number_input('Humidity')
wind_speed = st.number_input('Wind Speed')
precip = st.number_input('Precipitation (%)')
pressure = st.number_input('Atmospheric Pressure')
uv_index = st.number_input('UV Index')
visibility = st.number_input('Visibility (km)')
cloud_cover = st.selectbox('Cloud Cover', ['partly cloudy', 'clear', 'overcast', 'cloudy'])
season = st.selectbox('Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
location = st.selectbox('Location', ['inland', 'mountain', 'coastal'])

# Button to make prediction
if st.button('Predict Weather Type'):
    preprocessed_data = preprocess_input(temp, humidity, wind_speed, precip, pressure, uv_index, visibility, cloud_cover, season, location)
    prediction = model.predict(preprocessed_data)
    st.write(f'Predicted Weather Type: {prediction[0]}')
