# -*- coding: utf-8 -*-
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model (make sure fish_classifier_effnetb0.h5 is in the same folder)
model = load_model('fish_classifier_effnetb0.h5')

# List of class names
class_names = [
    'animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet', 'fish sea_food red_sea_bream',
    'fish sea_food sea_bass', 'fish sea_food shrimp',
    'fish sea_food striped_red_mullet', 'fish sea_food trout'
]

st.title("Fish Image Classification")

uploaded_file = st.file_uploader("Upload a fish image", type=['jpg', 'png'])

if uploaded_file is not None:
    # Load image with target size
    img = image.load_img(uploaded_file, target_size=(224, 224))

    # Convert to numpy array and normalize
    img_array = image.img_to_array(img) / 255.0

    # Add batch dimension for model input
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    # Show uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Get top 3 predictions
    top3_indices = predictions[0].argsort()[-3:][::-1]

    # Display top 3 predictions with confidence
    for i in top3_indices:
        st.write(f"{class_names[i]}: {predictions[0][i]*100:.2f}%")
