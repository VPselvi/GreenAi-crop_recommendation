# Importing necessary libraries
import streamlit as st
import pandas as pd
import joblib

# Load the data, model, label encoder, and scaler from the root directory
data = pd.read_csv('/Users/riyaz/Downloads/py project/GreenAi-crop_recommendation/crop_recommendation.csv')
model = joblib.load('/Users/riyaz/Downloads/py project/GreenAi-crop_recommendation/crop_recommendation_model.pkl')
label_encoder = joblib.load('/Users/riyaz/Downloads/py project/GreenAi-crop_recommendation/label_encoder.pkl')  # Load the label encoder
scaler = joblib.load('/Users/riyaz/Downloads/py project/GreenAi-crop_recommendation/scaler.pkl')  # Load the scaler

# Set the title for the app
st.title('Crop Recommendation System')
st.sidebar.header('User Input Parameters')

# Function to get user inputs
def user_input_features():
    Nitrogen = st.sidebar.number_input('Nitrogen')
    Phosphorus = st.sidebar.number_input('Phosphorus')
    Potassium = st.sidebar.number_input('Potassium')
    Temperature = st.sidebar.number_input('Temperature')
    Humidity = st.sidebar.number_input('Humidity')
    pH = st.sidebar.number_input('pH')
    Rainfall = st.sidebar.number_input('Rainfall')

    # Dictionary to hold user input data
    data = {
        'N': Nitrogen,
        'P': Phosphorus,
        'K': Potassium,
        'temperature': Temperature,
        'humidity': Humidity,
        'ph': pH,
        'rainfall': Rainfall
    }

    # Converting the dictionary into a DataFrame
    features = pd.DataFrame(data, index=[0])
    
    return features

# Fetch user input features and display them
input_df = user_input_features()
st.subheader('User Input Parameters')
st.write(input_df)

# Generate prediction based on user inputs
if st.button("Recommend Crop"):
    # Scale the input data using the scaler
    scaled_input = scaler.transform(input_df)
    
    # Get prediction (which will be a numeric label index)
    prediction_index = model.predict(scaled_input)[0]
    
    # Use LabelEncoder to convert the numeric prediction back to its original label
    predicted_crop = label_encoder.inverse_transform([prediction_index])[0]
    
    # Display the predicted crop (the actual label)
    st.subheader("Recommended Crop")
    st.write(predicted_crop)
