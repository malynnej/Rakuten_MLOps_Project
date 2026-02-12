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
st.subheader("MLOps Definition & Principles")
st.markdown("""
    **MLOps = Machine Learning** Operations as bridge between ML and daily operations   
    Managing full ML lifecycle from retrieving data, training models to deploying and monitoring them in real-world enviroment  
    * Build reproducible ML pipelines
    * Build reusable software environments
    * Managing and deploy models from anywhere
    * Monitor ML applications for operational and ML-related issues
    * Automation of end-to-end ML lifecycle 
    """)
image_ML = Image.open('./streamlit/images/MLPrinc.jpg')
st.image(image_ML, width=800)

st.write("---")

st.write("")
st.subheader("Basic MLOps Architecture Rakuten")
image_ML = Image.open('./streamlit/images/MLOps_Architecture.jpg')
st.image(image_ML, width=1000)
st.write("")