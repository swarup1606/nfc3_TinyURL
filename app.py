import os
import numpy as np
import pandas as pd
import requests
import json
import cv2
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

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

# Function to analyze soil health
def analyze_soil_health(pH, nitrogen, phosphorus, potassium, organic_matter):
    healthy = {'pH': (6.0, 7.5), 'nitrogen': (20, 50), 'phosphorus': (15, 40), 'potassium': (15, 40), 'organic_matter': (3, 6)}
    moderate = {'pH': (5.5, 6.0), 'nitrogen': (10, 20), 'phosphorus': (10, 15), 'potassium': (10, 15), 'organic_matter': (2, 3)}

    pH_status = 'Healthy' if healthy['pH'][0] <= pH <= healthy['pH'][1] else 'Moderate' if moderate['pH'][0] <= pH <= moderate['pH'][1] else 'Unhealthy'
    nitrogen_status = 'Healthy' if healthy['nitrogen'][0] <= nitrogen <= healthy['nitrogen'][1] else 'Moderate' if moderate['nitrogen'][0] <= nitrogen <= moderate['nitrogen'][1] else 'Unhealthy'
    phosphorus_status = 'Healthy' if healthy['phosphorus'][0] <= phosphorus <= healthy['phosphorus'][1] else 'Moderate' if moderate['phosphorus'][0] <= phosphorus <= moderate['phosphorus'][1] else 'Unhealthy'
    potassium_status = 'Healthy' if healthy['potassium'][0] <= potassium <= healthy['potassium'][1] else 'Moderate' if moderate['potassium'][0] <= potassium <= moderate['potassium'][1] else 'Unhealthy'
    organic_matter_status = 'Healthy' if healthy['organic_matter'][0] <= organic_matter <= healthy['organic_matter'][1] else 'Moderate' if moderate['organic_matter'][0] <= organic_matter <= moderate['organic_matter'][1] else 'Unhealthy'

    overall_health = {
        'pH': pH_status,
        'Nitrogen': nitrogen_status,
        'Phosphorus': phosphorus_status,
        'Potassium': potassium_status,
        'Organic Matter': organic_matter_status
    }

    return overall_health

# Function to process satellite images
def process_satellite_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Function to train the crop recommendation model
def train_crop_recommendation_model(soil_data, crop_production_data):
    merged_data = pd.merge(soil_data, crop_production_data, on='District')
    features = merged_data[['pH Level', 'Nitrogen Content (kg/ha)', 'Phosphorus Content (kg/ha)', 'Potassium Content (kg/ha)', 'Organic Matter (%)']]
    target = merged_data['Crop']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

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

# Function to create a simple CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Example for binary classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train a simple CNN model on synthetic data (placeholder)
def train_placeholder_model():
    model = create_cnn_model()
    
    X_train = np.random.rand(100, 64, 64, 3)  # 100 samples, 64x64 images, 3 channels (RGB)
    y_train = np.random.randint(2, size=100)  # Binary classification (0 or 1)
    y_train = to_categorical(y_train)  # One-hot encode labels

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save('crop_health_model.h5')
    return model

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        st.error("Error loading image. Please check the file path.")
        return None
    image = cv2.resize(image, (64, 64))  # Resize to model input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Function to predict crop health using the model
def predict_crop_health(model, image_path):
    image = preprocess_image(image_path)
    if image is not None:
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions)
        return "Healthy" if predicted_class == 0 else "Unhealthy"
    return None
import streamlit as st

# Function to add custom CSS
def add_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #000000; /* Black background for the main content */
            color: #ffffff; /* White text color */
        }
        .sidebar .sidebar-content {
            background-color: rgba(0, 0, 0, 0.7);
            color: #ffffff; /* White text color */
        }
        .css-1d391kg {
            background-color: rgba(0, 0, 0, 0.5);
            color: #ffffff; /* White text color */
        }
        .stButton>button {
            background-color: #f39c12;
            color: white;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #e67e22;
        }
        </style>
    """, unsafe_allow_html=True)

# Streamlit UI
def main():
    add_custom_css()

    st.title("AI-Based Smart Agriculture Assistant")

    # Sidebar menu for navigation
    menu = ["Home", "Soil Health Analysis", "Weather Forecasting", "Crop Recommendation", "Irrigation Management", "Crop Health Monitoring"]
    choice = st.sidebar.selectbox("Select Feature", menu)

    if choice == "Home":
        st.subheader("Welcome to the AI-Based Smart Agriculture Assistant")
        st.write("This system helps farmers optimize crop yield by analyzing soil conditions, weather patterns, and crop health monitoring.")

    elif choice == "Soil Health Analysis":
        st.subheader("Soil Health Analysis")
        st.write("Enter the soil parameters below to analyze the soil health:")
        pH = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
        nitrogen = st.number_input("Nitrogen Content (kg/ha)", min_value=0.0, value=20.0)
        phosphorus = st.number_input("Phosphorus Content (kg/ha)", min_value=0.0, value=15.0)
        potassium = st.number_input("Potassium Content (kg/ha)", min_value=0.0, value=15.0)
        organic_matter = st.number_input("Organic Matter (%)", min_value=0.0, max_value=100.0, value=5.0)
        
        if st.button("Analyze Soil Health"):
            health_status = analyze_soil_health(pH, nitrogen, phosphorus, potassium, organic_matter)
            st.write("Soil Health Status:")
            st.json(health_status)

    elif choice == "Weather Forecasting":
        st.subheader("Weather Forecasting")
        city = st.text_input("Enter city name", "London")
        if st.button("Get Weather Data"):
            weather_data = get_weather_data(city)
            if weather_data:
                st.write("Weather data:")
                temperature = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                weather_description = weather_data['weather'][0]['description'].capitalize()
                wind_speed = weather_data['wind']['speed']
                pressure = weather_data['main']['pressure']

                st.write(f"City: {city}")
                st.write(f"Temperature: {temperature}°C")
                st.write(f"Humidity: {humidity}%")
                st.write(f"Weather Description: {weather_description}")
                st.write(f"Wind Speed: {wind_speed} m/s")
                st.write(f"Pressure: {pressure} hPa")
            else:
                st.write("Failed to retrieve weather data.")

    elif choice == "Crop Recommendation":
        st.subheader("Crop Recommendation")
        st.write("Load soil and crop production data to train the model and get recommendations:")
        soil_data, crop_production_data = load_data()
        
        if soil_data is not None and crop_production_data is not None:
            st.write("Soil and crop production data loaded successfully.")
            if st.button("Train Crop Recommendation Model"):
                model = train_crop_recommendation_model(soil_data, crop_production_data)
                st.write("Crop recommendation model trained.")
            
            model = load_crop_recommendation_model()
            if model:
                st.write("Enter soil data to get crop recommendations:")
                pH = st.number_input("pH level", min_value=0.0, max_value=14.0, value=7.0)
                nitrogen = st.number_input("Nitrogen Content (kg/ha)", min_value=0.0, value=20.0)
                phosphorus = st.number_input("Phosphorus Content (kg/ha)", min_value=0.0, value=15.0)
                potassium = st.number_input("Potassium Content (kg/ha)", min_value=0.0, value=15.0)
                organic_matter = st.number_input("Organic Matter (%)", min_value=0.0, max_value=100.0, value=5.0)
                
                soil_data_row = [pH, nitrogen, phosphorus, potassium, organic_matter]
                recommended_crop = recommend_crops_with_model(model, soil_data_row)
                st.write(f"Recommended Crop: {recommended_crop}")

    elif choice == "Irrigation Management":
        st.subheader("Irrigation Management")
        city = st.text_input("Enter city name", "London")
        soil_moisture = st.number_input("Current soil moisture (%)", min_value=0.0, max_value=100.0, value=30.0)
        if st.button("Manage Irrigation"):
            weather_data1 = get_weather_data(city)
            irrigation_management(weather_data1, soil_moisture)

    elif choice == "Crop Health Monitoring":
        st.subheader("Crop Health Monitoring")
        
        uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_path = os.path.join("uploaded_images", uploaded_file.name)
            os.makedirs("uploaded_images", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(file_path, caption="Uploaded Image", use_column_width=True)
        
        uploaded_images = [f for f in os.listdir("uploaded_images") if f.endswith(('jpg', 'jpeg', 'png'))]
        selected_image = st.selectbox("Select an image for crop health monitoring", uploaded_images)
        
        if selected_image:
            st.write(f"Selected image: {selected_image}")
        
        if st.button("Train Placeholder Model"):
            model = train_placeholder_model()
            st.write("Placeholder model trained and saved as 'crop_health_model.h5'")
        
        model = None
        if os.path.isfile('crop_health_model.h5'):
            model = load_model('crop_health_model.h5')
            st.write("Model loaded successfully.")
        else:
            st.write("Model file not found. Please train the model first.")
        
        if st.button("Monitor Crop Health"):
            if model:
                if selected_image:
                    image_path = os.path.join("uploaded_images", selected_image)
                    health_status = predict_crop_health(model, image_path)
                    if health_status:
                        st.write(f"Crop Health Status: {health_status}")
                else:
                    st.write("No image selected. Please select an image to monitor crop health.")
            else:
                st.write("No model loaded. Please train or provide a model.")

if __name__ == "__main__":
    main()