import streamlit as st
import pandas as pd

# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="MLOps Project - Rakuten E-Commerce Product Classification",
    layout="wide"
)

st.write("")
st.write("Date: 13th February 2026")
st.write("Team: Julia Wilczoch, Steffen Karalus, Jenny Lam")
st.write("Mentor: Kilyan (Liora)")
st.write("")

st.title("Project Overview")

st.write("---")

# ========== THE CONTEXT ==========
st.subheader("Project Context")
st.write("This Project is based on the Rakuten France dataset and aims to classify e-commerce products into 27 categories using both text descriptions and product images.")
st.write("The dataset from Rakuten France contains 84,916 products with multilingual text descriptions (French, German, English, etc.) with associated images.")
st.write("See the following URL for more details: https://challengedata.ens.fr/challenges/35")
st.write("Main Goal is to build an MLOps architecture with a basic product classification model. Due to the focus on MLOps, we decided to limit the classification part only on the text base and excluded the image part for this project.")
st.write("---")

# ========== THE OBJECTIVE ==========
st.subheader("Project Objective")

st.write("[PLACEHOLDER - Alternative 1 to first objective: Optimize And Automate Development, Training And Deployment Of ML Model]")
st.write("[PLACEHOLDER - Alternative 2 to first objective: Efficient Deployment And Scalable Maintenance Of ML/DL Model]")

col1, col2, = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #d62728; padding: 20px; border-radius: 10px; text-align: center;">
        <p style="margin: 0; font-size: 20px; color: white;">Architecture</p>
        <h2 style="margin: 10px 0; font-size: 36px; color: white;">Reproducible End-To-End ML Model Architecture</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #8c564b; padding: 20px; border-radius: 10px; text-align: center;">
        <p style="margin: 0; font-size: 20px; color: white;">Data</p>
        <h5 style="margin: 10px 0; font-size: 48px; color: white;">Baseline ML Data Pipeline</h5>
        <p style="margin: 0; font-size: 14px; color: white;">based on text</p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Key metrics
st.subheader("Key Facts")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Microservices", "4")
col2.metric("MLOps Tools", "[PLACEHOLDER]")
col3.metric("Weighted F1 Score", "83%")

st.write("---")

# ========== LIMITATIONS ==========
st.subheader("Limitations")

with st.expander("**Text-only Classification Model**", expanded=False):
    st.write("""
    * the original Rakuten challenge includes text and image product data
    * only text classification here due to time and focus on MLOps
    """)

with st.expander("**Challenges Leading To Time Issues**", expanded=False):
    st.write("""
    * Deep Learning model leading to higher memory consumption and training time
    * faced hardware challenges, e.g. in building Docker images due to heavy packages
    * transferring an existing text classification model into MLOps context with microservices
    * limitations in uv workspaces (for using different setups of packages)
    """)

with st.expander("**Building MVP MLOps Architecture**", expanded=False):
    st.write("""
    * building basic MLOps Architecture with room for future improvement
    * focus on must-have features
    * working in one environment (no splitting of dev, staging, prod environment)
    """)

