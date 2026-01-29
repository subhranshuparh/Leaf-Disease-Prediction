import os
import json
from PIL import Image
import numpy as np
import streamlit as st

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import preprocess_input

# === Paths ===
working_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(working_dir, "trained_model", "leaf_vgg16_weights.weights.h5")
class_path = os.path.join(working_dir, "trained_model", "class_indices.json")

# === Load Classes ===
with open(class_path, "r") as f:
    mapping = json.load(f)

class_indices = {v: k for k, v in mapping.items()}
num_classes = len(class_indices)

# === Cache Model (IMPORTANT) ===
@st.cache_resource
def load_model():
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.load_weights(weights_path)
    return model

model = load_model()

# === Preprocessing ===
def preprocess_img(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

# === Prediction ===
def predict(image_file):
    arr = preprocess_img(image_file)
    preds = model.predict(arr, verbose=0)
    idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds)) * 100
    return class_indices[idx], confidence

# === Streamlit UI ===
st.title("🌱 Leaf Disease Prediction")

uploaded = st.file_uploader("Upload leaf image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Leaf Image", width=300)

    name, conf = predict(uploaded)

    st.success(f"Prediction: {name}")
    st.info(f"Confidence: {conf:.2f}%")
