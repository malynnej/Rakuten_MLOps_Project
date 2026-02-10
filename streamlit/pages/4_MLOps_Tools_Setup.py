import pandas as pd
import streamlit as st


# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="ML Ops Tools Setup",
    layout="wide"
)

st.title("ML Ops Tools Setup")

st.write("---")

# ==========GitHub Repository And UV Setup==========
st.subheader("GitHub Repository And UV Setup")
st.write("[PLACEHOLDER: Should we add this?]")

st.write("---")

# ==========Fast API==========
st.subheader("Fast API")
st.write("[PLACEHOLDER: Explain API Setup]")

st.write("---")

# ==========Docker Containerization==========
st.subheader("Docker Containerization")
st.write("[PLACEHOLDER: Explain Docker Setup]")

st.write("---")

# ==========Data And Model Versioning==========
st.subheader("Data And Model Versioning With DVC/Dagshub")
st.write("[PLACEHOLDER: Explain DVC/Dagshub Setup]")

st.write("---")

# ==========Experiment Tracking With MLFlow==========
st.subheader("Experiment Tracking With MLFlow")
st.write("[PLACEHOLDER: Explain MLFlow Setup]")

st.write("---")

# ==========Reverse Proxy And Load Balancing With Nginx==========
st.subheader("Reverse Proxy With Nginx")
st.write("[PLACEHOLDER: Explain Nginx Setup incl. security settings]")

st.write("---")

# ==========CI/CD==========
st.subheader("CI/CD")
st.write("[PLACEHOLDER: Explain CI/CD Setup with unit and integration tests, ruff actions, smoke tests]")
st.write("[PLACEHOLDER: Explain strategy of implementing CI/CD workflow, e.g. with Github Actions/Makefile (just theoretically)]")
st.write("---")

# ==========Retraining Workflow==========
st.subheader("Retraining Workflow")
st.write("[PLACEHOLDER: Explain strategy of Retraining Workflow (or move it to Data Pipeline page)]")
st.write("---")

# ==========Performance Monitoring With Prometheus/Grafana==========
st.subheader("Monitoring With Prometheus/Grafana")
st.write("[PLACEHOLDER: Explain Prometheus/Grafana Setup)]")
st.write("---")