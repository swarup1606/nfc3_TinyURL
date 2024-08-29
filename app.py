import streamlit as st
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to get weather data
def get_weather_data(city):
    api_key = "f853c70d8e3e9c8c1bf23747806c73a6"
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(base_url)
    data = response.json()
    return data

# Function to train the crop recommendation model
def train_crop_recommendation_model(soil_data, crop_production_data):
    # Check column names before merging
    st.write("Soil Data Columns:", soil_data.columns)
    st.write("Crop Production Data Columns:", crop_production_data.columns)

    # Merge the dataframes based on the correct key columns
    try:
        merged_data = pd.merge(soil_data, crop_production_data, on='District')
    except KeyError as e:
        st.error(f"KeyError: {e}. Please ensure both dataframes have a 'District' column.")
        return None
    
    # Features and target selection
    features = merged_data[['pH Level', 'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)', 'Potassium Content (kg/ha)']]
    target = merged_data['Crop']  
    
    # Splitting data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    # Training the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Saving the model
    joblib.dump(model, "crop_recommendation_model.pkl")
    st.write("Model training completed and saved.")
    
    return model

# Function to load the pre-trained model
def load_model():
    try:
        model = joblib.load("crop_recommendation_model.pkl")
        st.write("Model loaded successfully.")
        return model
    except FileNotFoundError:
        st.write("No trained model found. Please train a new model first.")
        return None

# Function to make crop recommendations
def recommend_crops_with_model(model, soil_data_row):
    prediction = model.predict([soil_data_row])
    return prediction[0]

# Function for irrigation management based on weather and soil data
def irrigation_management(weather_data):
    st.write("Irrigation Recommendations:")
    if weather_data['main']['humidity'] > 80:
        st.write("Low irrigation needed today due to high humidity.")
    else:
        st.write("Consider irrigating today as humidity is low.")

# Main application function
def main():
    st.title("AI-Based Farming Assistant")
    st.write("This system provides real-time insights and recommendations for optimizing crop yield.")
    
    # Weather forecast section
    st.header("Weather Forecast")
    city = st.text_input("Enter your city:", "London")
    if city:
        weather_data = get_weather_data(city)
        if weather_data and 'main' in weather_data:
            st.write(f"Weather in {city}:")
            st.json(weather_data)
        else:
            st.error("Failed to retrieve weather data. Please check the city name or try again later.")
    
    # Loading the uploaded data (simulating file input)
    st.header("Soil Health Analysis")
    
    # Load soil and crop production data
    soil_data = pd.read_csv('/mnt/data/soil_analysis_data.csv')
    crop_production_data = pd.read_csv('/mnt/data/crop_production_data.csv')
    
    st.write("Loaded soil and crop production data.")
    
    # Train the crop recommendation model
    if st.button("Train Crop Recommendation Model"):
        model = train_crop_recommendation_model(soil_data, crop_production_data)
    
    # Load the pre-trained model
    model = load_model()
    
    if model:
        # Crop recommendation section
        st.header("Crop Recommendation")
        soil_ph = st.number_input("Enter Soil pH:", value=6.5)
        soil_nitrogen = st.number_input("Enter Nitrogen Level:", value=20)
        soil_phosphorus = st.number_input("Enter Phosphorus Level:", value=20)
        soil_potassium = st.number_input("Enter Potassium Level:", value=20)
        soil_moisture = st.number_input("Enter Moisture Level:", value=30)
        
        # Prepare input for crop recommendation
        soil_data_row = [soil_ph, soil_nitrogen, soil_phosphorus, soil_potassium]
        
        if st.button("Recommend Crop"):
            recommended_crop = recommend_crops_with_model(model, soil_data_row)
            st.write(f"Recommended Crop: {recommended_crop}")
    
    # Irrigation management section
    st.header("Irrigation Management")
    if city and weather_data:
        irrigation_management(weather_data)

if __name__ == "_main_":
    main()