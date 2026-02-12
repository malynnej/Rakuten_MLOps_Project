import pandas as pd
import streamlit as st
from PIL import Image


# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="ML Ops Architecture",
    layout="wide"
)

st.title("ML Ops Architecture")

st.write("---")

# ========== ML Ops Architecture==========
st.subheader("MLOps Principles")
st.write("[PLACEHOLDER: Explain MLOps Principles]")

st.write("---")

st.write("")
st.subheader("Basic MLOps Architecture Rakuten")
image_ML = Image.open('./streamlit/images/MLOps_Architecture.jpg')
st.image(image_ML, width=1000)
st.write("")