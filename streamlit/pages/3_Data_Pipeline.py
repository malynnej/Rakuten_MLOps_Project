import pandas as pd
import streamlit as st
from PIL import Image


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
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Total Text Data",
        value="84,916"
    )

with col2:
    st.metric(
        label="Classes",
        value="27"
    )


challenge_col1, challenge_col2 = st.columns(2)

with challenge_col1:
    st.markdown("**Text Data Input:**")
    st.write("""
    - **Designation**: Product title
    - **Description**: Product description
    - **Product Type Code**: Product category (target variable)
    """)

with challenge_col2:
    st.markdown("**Text Exploration:**")
    st.write("""
    - **Multilingual text**: 30+ languages (60% French, 20% German, 20% others)
    - **Missing data**: 35% products have no description
    - **High text variance**: variance in descriptions from 0 till 20000 words
    - **HTML artifacts**: Descriptions contain HTML tags and special characters
    - **Severe Class Imbalance**: 13.4:1 ratio (largest vs smallest class)
    """)

st.markdown("**Class Imbalance**")
image_class = Image.open('./streamlit/images/Class_distribution.png')
st.image(image_class, width=1000)

st.write("---")

# ========== Data Pipeline==========
st.subheader("Data Pipeline")

st.write("")
image_DP = Image.open('./streamlit/images/Datapipe_v2.jpg')
st.image(image_DP, width=1100)
st.write("")

st.write("---")
# ==========Performance Metrics==========
st.subheader("Performance Metrics")

st.write("")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Base Model",
        value="bert-base-uncased"
    )

with col2:
    st.metric(
        label="Epochs",
        value="5"
    )

with col3:
    st.metric(
        label="LR",
        value="5.0e-5"
    )

with col4:
    st.metric(
        label="Dropout",
        value="0.1"
    )

st.write("")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #50b63e; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Weighted F1</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">83%</h2>
        <p style="margin: 0; font-size: 16px; color: white;">Overall Metric</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #7e7e7e; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Validation Accuracy</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">83%</h2>
        <p style="margin: 0; font-size: 16px; color: white;"></p>
    </div>
    """, unsafe_allow_html=True)
    

with col2:

    st.markdown("""
    <div style="background-color: #8fce00; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Test Accuracy</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">83%</h2>
        <p style="margin: 0; font-size: 16px; color: white;"></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #7e7e7e; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Validation Loss</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">0.58</h2>
        <p style="margin: 0; font-size: 16px; color: white;"></p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    image_result_1 = Image.open('./streamlit/images/f1_scores_by_class.png')
    st.image(image_result_1, caption='F1 Scores By Class', width="content")

with row1_col2:
    image_result_3 = Image.open('./streamlit/images/confusion_matrix.png')
    st.image(image_result_3, caption='Confusion Matrix', width="content")

st.write("")