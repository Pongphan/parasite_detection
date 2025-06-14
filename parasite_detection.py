import streamlit as st
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

st.title("AI Detector")
st.subheader("Upload & View Image")
st.write("Upload an image and view it below.")

model_path = "model/aug_img_cnn.h5"
model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

#------------------------------------------------------------------------------
# ObjectDet function
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "tif"])
if uploaded_file is not None:
    try:
        image = np.array(Image.open(uploaded_file))
        st.write(f"Image successfully loaded!: {uploaded_file.name}")
        
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        st.image(image, caption="Uploaded Image")

        output_img = ObjectDet(image, 0.90, 0.3, 0.5)
        st.image(output_img, caption="Processed Image")

    except Exception as e:
        st.error(f"Error loading image: {e}")
