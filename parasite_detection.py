import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load your models (example with 4 classes)
model1 = load_model('ev_cnn_mobile.keras')

models = [model1]
class_names = ['Class 0', 'Class 1']
patch_sizes = 650, 650

def search_patches(img):
    h, w, _ = img.shape
    results = []
    for idx, (ph, pw) in enumerate(patch_sizes):
        for y in range(0, h-ph, ph//2):
            for x in range(0, w-pw, pw//2):
                patch = img[y:y+ph, x:x+pw]
                patch_resized = cv2.resize(patch, (ph, pw))
                patch_input = patch_resized / 255.0
                patch_input = np.expand_dims(patch_input, axis=0)
                pred = models[idx].predict(patch_input)
                if pred[0, 0] > 0.9:  # Adjust threshold as needed
                    results.append((class_names[idx], x, y, pw, ph))
    return results

st.title("Patch Searching Object Detection App")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Searching for objects...")
    boxes = search_patches(img)
    if boxes:
        for label, x, y, w, h in boxes:
            st.write(f"Found {label} at [{x}, {y}, {w}, {h}]")
    else:
        st.write("No objects detected.")
