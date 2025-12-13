import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('C:/Users/PMYLS/Downloads/pneumonia_model.h5')
st.title("Pneumonia X-ray Detection")
st.write("Upload a chest X-ray image and the model will predict Pneumonia or Normal")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]
    result = "Pneumonia" if prediction > 0.5 else "Normal"

    st.write(f"Predicted class: **{result}** (Probability: {prediction:.2f})")
