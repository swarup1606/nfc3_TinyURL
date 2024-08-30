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

    if data.get('cod') != 200:
        st.error(f"Error retrieving weather data for {city}: {data.get('message')}")
        return None
    
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    precipitation = data.get('rain', {}).get('1h', 0)
    wind_speed = data['wind']['speed']
    
    st.write(f"Temperature: {temperature}°C / तापमान: {temperature}°C")
    st.write(f"Humidity: {humidity}% / आर्द्रता: {humidity}%")
    st.write(f"Precipitation: {precipitation} mm / वर्षा: {precipitation} मिमी")
    st.write(f"Wind Speed: {wind_speed} m/s / पवन की गति: {wind_speed} मी/से")

    return data


# Function to load CSV data
def load_data():
    try:
        soil_data = pd.read_csv("soil_analysis_data.csv")
        crop_production_data = pd.read_csv("crop_production_data.csv")
        return soil_data, crop_production_data
    except Exception as e:
        st.error(f"Error loading data: {e} / डेटा लोड करने में त्रुटि: {e}")
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
    st.write(f"Model Accuracy: {accuracy * 100:.2f}% / मॉडल की सटीकता: {accuracy * 100:.2f}%")

    joblib.dump(model, "crop_recommendation_model.pkl")
    st.write("Crop recommendation model trained and saved. / फसल सिफारिश मॉडल प्रशिक्षित और सहेजा गया।")
    return model

# Function to load the trained crop recommendation model
def load_crop_recommendation_model():
    try:
        model = joblib.load("crop_recommendation_model.pkl")
        st.write("Crop recommendation model loaded successfully. / फसल सिफारिश मॉडल सफलतापूर्वक लोड किया गया।")
        return model
    except FileNotFoundError:
        st.write("No trained crop recommendation model found. Train a new model first. / कोई प्रशिक्षित फसल सिफारिश मॉडल नहीं मिला। पहले एक नया मॉडल प्रशिक्षित करें।")
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
    st.write("Irrigation model trained and saved. / सिंचाई मॉडल प्रशिक्षित और सहेजा गया।")
    return model

# Function to load the irrigation model
def load_irrigation_model():
    try:
        model = joblib.load("irrigation_model.pkl")
        st.write("Irrigation model loaded successfully. / सिंचाई मॉडल सफलतापूर्वक लोड किया गया।")
        return model
    except FileNotFoundError:
        st.write("No trained irrigation model found. Train a new model first. / कोई प्रशिक्षित सिंचाई मॉडल नहीं मिला। पहले एक नया मॉडल प्रशिक्षित करें।")
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

        st.write(f"Current Soil Moisture: {soil_moisture}% / वर्तमान मिट्टी की नमी: {soil_moisture}%")
        st.write(f"Predicted Soil Moisture: {predicted_soil_moisture:.2f}% / भविष्यवाणी की गई मिट्टी की नमी: {predicted_soil_moisture:.2f}%")

        if soil_moisture < predicted_soil_moisture:
            st.write("Irrigation needed to reach optimal soil moisture levels. / आदर्श मिट्टी की नमी स्तर तक पहुंचने के लिए सिंचाई की आवश्यकता है।")
        else:
            st.write("Soil moisture is sufficient; no irrigation needed. / मिट्टी की नमी पर्याप्त है; कोई सिंचाई की आवश्यकता नहीं है।")
    else:
        st.write("Unable to perform irrigation management without a trained model. / प्रशिक्षित मॉडल के बिना सिंचाई प्रबंधन नहीं किया जा सकता।")

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

# Function to load the CNN model for crop health monitoring
def load_cnn_model():
    try:
        model = load_model("crop_health_model.h5")
        st.write("Crop health CNN model loaded successfully. / फसल स्वास्थ्य CNN मॉडल सफलतापूर्वक लोड किया गया।")
        return model
    except FileNotFoundError:
        st.write("No trained crop health model found. Train a new model first. / कोई प्रशिक्षित फसल स्वास्थ्य मॉडल नहीं मिला। पहले एक नया मॉडल प्रशिक्षित करें।")
        return None

# Function to predict crop health
def predict_crop_health(cnn_model, image):
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    prediction = cnn_model.predict(image)
    return prediction

# Function to get advice based on problem headings
def get_advice_for_heading(heading):
    advice = {
        'Pest Infestation': "Regularly inspect plants for pests and use appropriate pesticides. / पौधों की नियमित जांच करें और उचित कीटनाशक का उपयोग करें।",
        'Nutrient Deficiency': "Apply a balanced fertilizer to address nutrient deficiencies. / पोषक तत्वों की कमी को दूर करने के लिए एक संतुलित उर्वरक का उपयोग करें।",
        'Watering Issues': "Ensure plants receive adequate water based on weather conditions. / सुनिश्चित करें कि पौधों को मौसम की परिस्थितियों के आधार पर पर्याप्त पानी मिले।",
        'Soil Quality': "Improve soil quality by adding organic matter and adjusting pH levels. / मिट्टी की गुणवत्ता को बेहतर बनाने के लिए जैविक पदार्थ जोड़ें और pH स्तर को समायोजित करें।",
        'Crop Diseases': "Identify and treat crop diseases promptly with suitable methods. / फसल की बीमारियों की पहचान करें और उपयुक्त विधियों से तुरंत उपचार करें।"
    }
    return advice.get(heading, "No advice available for this issue. / इस समस्या के लिए कोई सलाह उपलब्ध नहीं है।")

# Streamlit app
def main():
    st.title("AI-Powered Smart Agriculture Assistant")
    st.sidebar.title("Features")

    features = ["Home / घर", "Weather Forecast / मौसम पूर्वानुमान", "Soil Health Analysis / मिट्टी स्वास्थ्य विश्लेषण", 
                "Crop Recommendation / फसल सिफारिश", "Irrigation Management / सिंचाई प्रबंधन", 
                "Crop Health Monitoring / फसल स्वास्थ्य निगरानी", "Article Advice / लेख सलाह"]
    
    selection = st.sidebar.radio("Select a feature / एक सुविधा चुनें", features)

    if selection == "Home / घर":
        st.header("Welcome to the Smart Agriculture Assistant \n स्मार्ट कृषि सहायक में आपका स्वागत है")
        st.write("This application helps farmers optimize crop yield through various AI-powered features.\n  \nयह एप्लिकेशन किसानों को विभिन्न एआई-संचालित सुविधाओं के माध्यम से फसल उत्पादन को अनुकूलित करने में मदद करता है।")

    elif selection == "Weather Forecast / मौसम पूर्वानुमान":
        city = st.text_input("Enter city name / शहर का नाम दर्ज करें")
        if st.button("Get Weather / मौसम प्राप्त करें"):
            if city:
                data = get_weather_data(city)
                st.write(data)
            else:
                st.write("Please enter a city name. / कृपया एक शहर का नाम दर्ज करें।")

    elif selection == "Soil Health Analysis / मिट्टी स्वास्थ्य विश्लेषण":
        st.header("Soil Health Analysis / मिट्टी स्वास्थ्य विश्लेषण")
        pH = st.number_input("Enter pH level / pH स्तर दर्ज करें")
        nitrogen = st.number_input("Enter Nitrogen content (kg/ha) / नाइट्रोजन सामग्री (किलोग्राम/हेक्टेयर) दर्ज करें")
        phosphorus = st.number_input("Enter Phosphorus content (kg/ha) / फास्फोरस सामग्री (किलोग्राम/हेक्टेयर) दर्ज करें")
        potassium = st.number_input("Enter Potassium content (kg/ha) / पोटेशियम सामग्री (किलोग्राम/हेक्टेयर) दर्ज करें")
        organic_matter = st.number_input("Enter Organic Matter (%) / कार्बनिक पदार्थ (%) दर्ज करें")
        
        if st.button("Analyze Soil Health / मिट्टी के स्वास्थ्य का विश्लेषण करें"):
            health = analyze_soil_health(pH, nitrogen, phosphorus, potassium, organic_matter)
            st.write(health)

    elif selection == "Crop Recommendation / फसल सिफारिश":
        soil_data, crop_production_data = load_data()
        if soil_data is not None and crop_production_data is not None:
            model = load_crop_recommendation_model()
            if model is None:
                model = train_crop_recommendation_model(soil_data, crop_production_data)
            pH = st.number_input("Enter pH level / pH स्तर दर्ज करें")
            nitrogen = st.number_input("Enter Nitrogen content (kg/ha) / नाइट्रोजन सामग्री (किलोग्राम/हेक्टेयर) दर्ज करें")
            phosphorus = st.number_input("Enter Phosphorus content (kg/ha) / फास्फोरस सामग्री (किलोग्राम/हेक्टेयर) दर्ज करें")
            potassium = st.number_input("Enter Potassium content (kg/ha) / पोटेशियम सामग्री (किलोग्राम/हेक्टेयर) दर्ज करें")
            organic_matter = st.number_input("Enter Organic Matter (%) / कार्बनिक पदार्थ (%) दर्ज करें")
            
            if st.button("Recommend Crops / फसल की सिफारिश करें"):
                prediction = recommend_crops_with_model(model, [pH, nitrogen, phosphorus, potassium, organic_matter])
                st.write(f"Recommended Crop: {prediction} / अनुशंसित फसल: {prediction}")

    elif selection == "Irrigation Management / सिंचाई प्रबंधन":
        weather_city = st.text_input("Enter city name for weather forecast / मौसम पूर्वानुमान के लिए शहर का नाम दर्ज करें")
        soil_moisture = st.number_input("Enter current soil moisture (%) / वर्तमान मिट्टी की नमी (%) दर्ज करें")
        
        if st.button("Manage Irrigation / सिंचाई प्रबंधन"):
            if weather_city:
                weather_data = get_weather_data(weather_city)
                irrigation_management(weather_data, soil_moisture)
            else:
                st.write("Please enter a city name. / कृपया एक शहर का नाम दर्ज करें।")

    elif selection == "Crop Health Monitoring / फसल स्वास्थ्य निगरानी":
        uploaded_file = st.file_uploader("Choose an image of the crop / फसल की छवि चुनें", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            cnn_model = load_cnn_model()
            if cnn_model:
                prediction = predict_crop_health(cnn_model, image)
                st.write(f"Prediction: {prediction} / भविष्यवाणी: {prediction}")

    elif selection == "Article Advice / लेख सलाह":
        article_headings = ['Pest Infestation / कीटों का संक्रमण', 'Nutrient Deficiency / पोषक तत्वों की कमी', 
                            'Watering Issues / पानी की समस्याएँ', 'Soil Quality / मिट्टी की गुणवत्ता', 
                            'Crop Diseases / फसल की बीमारियाँ']
        
        heading = st.selectbox("Select an article heading / लेख शीर्षक चुनें", article_headings)
        if st.button("Get Advice / सलाह प्राप्त करें"):
            advice = get_advice_for_heading(heading)
            st.write(advice)

if __name__ == "__main__":
    main()
