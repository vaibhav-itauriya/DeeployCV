import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('deepfake_model.h5')

# Streamlit app interface
st.title("Deepfake Detection")
st.write("Upload an image to determine whether it's real or fake.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    result = "Fake" if prediction[0] > 0.5 else "Real"

    # Display result
    st.write(f"Prediction: **{result}**")
