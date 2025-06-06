import streamlit as st

st.title("AI Detector")
st.subheader("Upload & View Image")
st.write("Upload an image and view it below.")

name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")

age = st.slider("How old are you?", 0, 100, 25)
st.write(f"Your age: {age}")

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file:
    st.image(uploaded_file)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [10, 20, 15, 25, 30]
})

st.write("Sample DataFrame", df)

plt.plot(df["x"], df["y"])
st.pyplot(plt)

st.title("Contact Form")

with st.form("contact"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    message = st.text_area("Message")
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.success(f"Thanks, {name}! We received your message.")
