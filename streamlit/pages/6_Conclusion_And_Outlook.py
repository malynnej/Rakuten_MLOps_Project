import pandas as pd
import streamlit as st


# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="Conclusion And Outlook",
    layout="wide"
)

st.title("Conclusion And Outlook")

st.write("---")

# ========== KEY FINDINGS ==========
st.subheader("Key Findings")

st.write("[PLACEHOLDER]")

st.write("---")

# ========== OUTLOOK ==========
st.subheader("Outlook")

st.markdown("""
    * automate CI/CD with workflow
    * automate retraining workflow
    * decrease Docker images with lighter pytorch or other approaches
    * having an automated retraining workflow
    * security improvements
    * add image classification and fusion model to the architecture
    * add more performance monitoring metrics
    * enhance setup of tools: e.g. load balancing with Nginx, workflow orchestration with Airflow
    * having dev/stage/prod environment
    * add model registry and serving with MLFlow
    * Automation of end-to-end ML lifecycle 
    """)
