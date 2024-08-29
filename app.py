import streamlit as st
import pandas as pd
import requests
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Function to get weather data from API
def get_weather_data(city):
    api_key = "f853c70d8e3e9c8c1bf23747806c73a6"
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()
    return data

# Function to load CSV data
def load_data():
    try:
        soil_data = pd.read_csv("soil_analysis_data.csv")
        crop_production_data = pd.read_csv("crop_production_data.csv")
        return soil_data, crop_production_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to train the crop recommendation model
def train_crop_recommendation_model(soil_data, crop_production_data):
    # Merge datasets on the 'District' column
    merged_data = pd.merge(soil_data, crop_production_data, on='District')
    
    # Prepare the data
    features = merged_data[['pH Level', 'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)', 'Potassium Content (kg/ha)', 'Organic Matter (%)']]
    target = merged_data['Crop']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model
    joblib.dump(model, "crop_recommendation_model.pkl")
    st.write("Crop recommendation model trained and saved.")
    return model

# Function to load the trained crop recommendation model
def load_crop_recommendation_model():
    try:
        model = joblib.load("crop_recommendation_model.pkl")
        st.write("Crop recommendation model loaded successfully.")
        return model
    except FileNotFoundError:
        st.write("No trained crop recommendation model found. Train a new model first.")
        return None

# Function to recommend crops based on soil data using the trained model
def recommend_crops_with_model(model, soil_data_row):
    prediction = model.predict([soil_data_row])
    return prediction[0]

# Function to get historical weather data for training the irrigation model
def get_historical_weather_data():
    return pd.DataFrame({
        'temperature': [22, 24, 20, 23, 25],
        'humidity': [60, 65, 70, 55, 50],
        'precipitation': [5, 0, 10, 0, 0],
        'soil_moisture': [30, 28, 35, 33, 30]
    })

# Function to train the irrigation model
def train_irrigation_model():
    data = get_historical_weather_data()
    X = data[['temperature', 'humidity', 'precipitation']]
    y = data['soil_moisture']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "irrigation_model.pkl")
    st.write("Irrigation model trained and saved.")
    return model

# Function to load the irrigation model
def load_irrigation_model():
    try:
        model = joblib.load("irrigation_model.pkl")
        st.write("Irrigation model loaded successfully.")
        return model
    except FileNotFoundError:
        st.write("No trained irrigation model found. Train a new model first.")
        return None

# Function for irrigation management with predictive analytics
def irrigation_management(weather_data, soil_moisture):
    model = load_irrigation_model()
    
    if model:
        temp = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        precipitation = weather_data.get('rain', {}).get('1h', 0)

        prediction = model.predict([[temp, humidity, precipitation]])
        predicted_soil_moisture = prediction[0]

        st.write(f"Current Soil Moisture: {soil_moisture}%")
        st.write(f"Predicted Soil Moisture: {predicted_soil_moisture:.2f}%")

        if soil_moisture < predicted_soil_moisture:
            st.write("Irrigation needed to reach optimal soil moisture levels.")
        else:
            st.write("Soil moisture is sufficient; no irrigation needed.")
    else:
        st.write("Unable to perform irrigation management without a trained model.")

# Streamlit UI
def main():
    st.title("AI-Based Farming Assistant")
    st.write("This system provides real-time insights and recommendations for optimizing crop yield.")

    # Section for Weather Forecast
    st.header("Weather Forecast")
    city = st.text_input("Enter your city:", "London")
    if city:
        weather_data = get_weather_data(city)
        st.write(f"Weather in {city}:")
        st.json(weather_data)

        # Show weather data in text format
        st.write("Weather Details:")
        st.write(f"Temperature: {weather_data['main']['temp']}°C")
        st.write(f"Humidity: {weather_data['main']['humidity']}%")
        st.write(f"Precipitation: {weather_data.get('rain', {}).get('1h', 0)} mm")

    # Load data
    soil_data, crop_production_data = load_data()

    if soil_data is None or crop_production_data is None:
        st.stop()

    # Train irrigation model if required
    if st.button("Train Irrigation Model"):
        train_irrigation_model()

    # Check if model is loaded and perform irrigation management
    if weather_data:
        st.header("Irrigation Management")
        soil_moisture = st.number_input("Enter Current Soil Moisture Level (%):", value=30)
        irrigation_management(weather_data, soil_moisture)

    # Train crop recommendation model if required
    if st.button("Train Crop Recommendation Model"):
        model = train_crop_recommendation_model(soil_data, crop_production_data)

    # Load the model
    model = load_crop_recommendation_model()

    if model:
        # Get soil data input from user for crop recommendation
        st.header("Crop Recommendation")
        soil_ph = st.number_input("Enter Soil pH:", value=6.5)
        soil_nitrogen = st.number_input("Enter Nitrogen Level:", value=20)
        soil_phosphorus = st.number_input("Enter Phosphorus Level:", value=20)
        soil_potassium = st.number_input("Enter Potassium Level:", value=20)
        organic_matter = st.number_input("Enter Organic Matter Level (%):", value=5)

        # Create a data row for the prediction
        soil_data_row = [soil_ph, soil_nitrogen, soil_phosphorus, soil_potassium, organic_matter]

        if st.button("Recommend Crop"):
            recommended_crop = recommend_crops_with_model(model, soil_data_row)
            st.write(f"Recommended Crop: {recommended_crop}")

if __name__ == "_main_":
    main()