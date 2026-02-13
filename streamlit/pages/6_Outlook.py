import streamlit as st


# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="Conclusion And Outlook",
    layout="wide"
)

st.title("Outlook")

st.write("---")

st.subheader("Datapipe")

st.markdown("""
    * automate retraining workflow
    * use lighter pytorch or other approaches to decrease Docker images
    * add model registry and serving with MLFlow
    * add image classification and fusion model to the architecture
    """)

st.subheader("Further MLOps Setup")

st.markdown("""
    * automate CI/CD with workflow
    * security improvements (API security, user management)
    * add more performance monitoring metrics (model metrics, drift monitoring)
    * enhance setup: e.g. load balancing with Nginx, workflow orchestration with Airflow, scalability with Kubernetes
    * having dev/stage/prod environment
    """)



