import streamlit as st
import streamlit.components.v1 as components

st.title("API Documentation")

# Dropdown to select API
api_selection = st.selectbox(
    "Select API Documentation:",
    ["Data API", "Train API", "Predict API", "Evaluate API"]
)

# Map selection to URLs
api_urls = {
    "Data API": "https://localhost:8443/data/docs",
    "Train API": "https://localhost:8443/train/docs",
    "Predict API": "https://localhost:8443/predict/docs",
    "Evaluate API": "https://localhost:8443/evaluate/docs"
}

# Display the selected API docs
st.markdown(f"### {api_selection} Documentation")

# Show iframe with selected URL
components.iframe(
    api_urls[api_selection],
    height=800,
    scrolling=True
)
