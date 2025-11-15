import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("mnist_model.h5")
st.title(" Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9) and the model will predict it.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    if img.mean() > 127:
        img = 255 - img
    img = img.astype('float32') / 255.0
    
    img = img.reshape(1, 28, 28)
    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.success(f"Predicted Digit: {digit} (Confidence: {confidence:.2f})")
