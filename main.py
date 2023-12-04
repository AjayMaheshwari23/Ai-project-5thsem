import os
import cv2
import time
import pickle
import tempfile
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ReLU

# Load your trained model from a pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to make predictions
def predict_image(img):
    prediction = 5436
    resized_img = cv2.resize(img, (1080,720) )
    cur_pred = []
    cur_pred.append(resized_img)
    cur_pred = np.array(cur_pred)

    print(cur_pred.shape)
    predicted_labels = ( model.predict(cur_pred) >= 0.5).astype('int64')

    #predicted_labels.shape
    print(predicted_labels)
    print(resized_img.shape)
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
    prediction = predict_image(img)

    # # Display prediction
    # st.write("Model Prediction:")
    # st.write(prediction)



def get_cap(device_num):
    cap = cv2.VideoCapture(device_num)
    return cap

def save_frame(device_num, cycle):
    cap = get_cap(device_num)

    if not cap.isOpened():
        st.error("Error: Unable to open the webcam.")
        return

    n = 0
    placeholder = st.empty()  # Create an empty placeholder


    st.title('Dynamic Placeholder Text')
    placeholder_text = st.empty()

    while n <= cycle:
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        placeholder.image(rgb_frame, channels="RGB", use_column_width=True, caption=f"Frame {n}")

        updated_text =  predict_image(frame)
        placeholder_text.text(updated_text)

        # Update every second
        time.sleep(1)

        n += 1

    # Release the webcam
    cap.release()

# Streamlit app
st.title("Webcam Stream in Streamlit")
save_frame(0, 60)
