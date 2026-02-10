import pandas as pd
import streamlit as st


# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="Data Pipeline",
    layout="wide"
)

st.title("Data Pipeline")

st.write("---")

# ========== Data Exploration==========
st.subheader("Data Exploration")
st.write("[PLACEHOLDER: Simple Data Context with amount of classes, data, target variable, class imbalance]")

st.write("---")

# ========== Data Pipeline==========
st.subheader("Data Pipeline")

st.write("")
st.write("[PLACEHOLDER: Show Data Pipeline from data input to output With Microservices]")
st.write("[PLACEHOLDER: Optionally include BERT architecture]")
st.write("")

# ==========Performance Metrics==========
st.subheader("Performance Metrics")

st.write("")
st.write("[PLACEHOLDER: Show performance metrics based on set parameter - accuracy, weighted F1, conf. matrix, training performance]")
st.write("")