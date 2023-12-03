import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ReLU
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import tempfile

# Load your trained model from a pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to make predictions
def predict_image(img):
    prediction = 5436
    resized_img = cv2.resize(img, (96,96))
    resized_img = resized_img / 255.0  # Normalize pixel values (adjust based on your model training)
    print(resized_img.shape)
    # input_data = np.expand_dims(resized_img, axis=0)
    #print(resized_img.shape)
    predicted_labels = (model.predict(np.array([resized_img])) >= 0.5).astype('int64').flatten()
    print(predicted_labels.shape)
    # prediction = model.predict(input_data)
    # prediction  = (model.predict(np.array(input_data)) >= 0.5).astype('int64').flatten()
    # print(prediction.shape)
    return predicted_labels[0]

# Streamlit UI
st.title("Image Prediction App")

# File Upload Widget
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    #Read the uploaded image
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close()

    # Use OpenCV to read the image
    img = cv2.imread(temp_file_path)
    # img = cv2.imread(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    #print(img.shape)
    # # Display the uploaded image
    #st.image(img, caption="Uploaded Image", use_column_width=True)

    # # Make predictions
    prediction = predict_image(img)

    # # Display predictions
    st.write("Model Prediction:")
    st.write(prediction)
