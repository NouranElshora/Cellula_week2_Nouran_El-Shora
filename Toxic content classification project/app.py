import streamlit as st
import pandas as pd
import os
from datetime import datetime
from PIL import Image

from imagecaption import generate_caption
from classifier import classify_text


DATABASE_FILE = "cellula toxic data.csv"

st.title("Multimodal Toxic Content Classification System")

st.write("Upload text or image to detect toxicity category.")

option = st.radio("Select Input Type:", ("Text", "Image"))

if option == "Text":

    user_text = st.text_area("Enter text")

    if st.button("Classify"):

        if user_text.strip() != "":

            result = classify_text(user_text)

            st.success(f"Predicted Category: {result}")

            new_row = {
                "query": user_text,
                "image descriptions": "",
                "Toxic Category": result
            }

            df = pd.read_csv(DATABASE_FILE)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATABASE_FILE, index=False)


elif option == "Image":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image)

        if st.button("Generate & Classify"):

            caption = generate_caption(image)

            st.write(f"Generated Caption: {caption}")

            result = classify_text(caption)

            st.success(f"Predicted Category: {result}")

            new_row = {
                "query": "",
                "image descriptions": caption,
                "Toxic Category": result
            }

            df = pd.read_csv(DATABASE_FILE)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(DATABASE_FILE, index=False)


st.subheader("View Database")

if st.button("Show All Records"):
    df = pd.read_csv(DATABASE_FILE)
    st.dataframe(df)
