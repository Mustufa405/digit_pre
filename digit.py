import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from PIL import Image

st.title("Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0–9) and the model will predict it.")
st.subheader("Upload Image")

if "model" not in st.session_state:
    st.write("⚙️ Training model for first use... Please wait!")
    (train_images, train_labels), _ = mnist.load_data()
    train_images = train_images / 255.0
    train_labels = to_categorical(train_labels)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=3, batch_size=32, verbose=0)
    st.session_state.model = model
    st.success("Model Ready You can upload digits now")

uploaded_file = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])  
if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((28, 28))  
    st.image(img, caption="Uploaded Digit", width=150)
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1, 28, 28)

    prediction = st.session_state.model.predict(img_arr)
    digit = np.argmax(prediction)

    st.success(f"🎉 Predicted Digit: {digit}")

