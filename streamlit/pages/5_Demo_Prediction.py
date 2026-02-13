"""
Live Prediction Page - Streamlit Integration
"""

import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
from datetime import datetime
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="Demo Prediction",
    layout="wide"
)

st.title("Live Product Category Prediction")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'api_url' not in st.session_state:
    st.session_state.api_url = "https://localhost:8443/predict/"
if 'auth' not in st.session_state:
    st.session_state.auth = None


# Login Form
def login_page():
    st.markdown("### Set Credentials to Access Prediction Service")
    
    with st.form("login_form"):
        api_url = st.text_input("API URL", value="https://localhost:8443/predict/")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Set and verify credentials", type="primary", use_container_width=True)
        
        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                # Test authentication
                try:
                    auth = HTTPBasicAuth(username, password)
                    response = requests.get(
                        f"{api_url}",
                        auth=auth,
                        verify=False,
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.api_url = api_url
                        st.session_state.auth = auth
                        st.session_state.username = username
                        st.success("Login successful")
                        st.rerun()
                    elif response.status_code == 401:
                        st.error("Invalid username or password")
                    else:
                        st.error(f"API Error: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error(f"Cannot connect to {api_url}. Check if API is running and URL is correct.")
                except requests.exceptions.Timeout:
                    st.error("Connection timeout. API is not responding.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Logout button in sidebar
def show_logout():
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.get('username', 'User')}")
        st.markdown(f"**API:** {st.session_state.api_url}")
        if st.button("Change user", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.auth = None
            st.rerun()

# Check API health
def check_api_health():
    try:
        response = requests.get(
            f"{st.session_state.api_url}health",
            auth=st.session_state.auth,
            verify=False,  # Skip SSL verification for self-signed cert
            timeout=5
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def prediction_page():
    show_logout()

    # Prediction mode selection
    st.markdown("---")
    prediction_mode = st.radio(
        "Select Prediction Mode:",
        ["Single Text", "Product (Designation + Description)", "Batch Prediction"],
        horizontal=True
    )

    # Display API status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### API Status")
    with col2:
        if check_api_health():
            st.success("Online")
        else:
            st.error("Offline")
            st.stop()

    # Single Text Prediction
    if prediction_mode == "Single Text":
        st.markdown("### Single Text Prediction: `/predict_text`")
        
        text_input = st.text_area(
            "Enter product text:",
            placeholder="e.g., Nike Air Max 90 running shoes black size 42",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            return_probs = st.checkbox("Show probabilities", value=True)
        with col2:
            top_k = st.slider("Top predictions", min_value=1, max_value=10, value=3)
        
        if st.button("Predict", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("Please enter some text")
            else:
                with st.spinner("Making prediction..."):
                    try:
                        payload = {
                            "text": text_input,
                            "return_probabilities": return_probs,
                            "top_k": top_k
                        }
                        
                        response = requests.post(
                            f"{st.session_state.api_url}predict_text", 
                            json=payload,
                            auth=st.session_state.auth,
                            verify=False,  
                            timeout=120
                        )

                        # Show request
                        with st.expander("View Request Payload"):
                            st.json(payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("Prediction Complete")
                            
                            # Display primary prediction
                            st.markdown("#### Primary Prediction")
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Category", result.get("predicted_category", "N/A"))
                            with col2:
                                if "confidence" in result:
                                    st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            # Display top-k predictions
                            if "top_predictions" in result:
                                st.markdown("#### Top Predictions")
                                
                                for i, pred in enumerate(result["top_predictions"], 1):
                                    class_id = pred.get("category", "N/A")
                                    prob = pred.get("probability", 0)
                                    
                                    st.markdown(f"**{i}. Class {class_id}**")
                                    st.progress(prob)
                                    st.caption(f"Probability: {prob:.4f} ({prob*100:.2f}%)")
                            
                            # Show raw response
                            with st.expander("View Raw API Response"):
                                st.json(result)
                        
                        elif response.status_code == 401:
                            st.error("Authentication failed. Check username and password.")

                        else:
                            st.error(f"API Error: {response.status_code}")
                            st.code(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API through Nginx. Check if services are running.")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The model may be processing a large batch.")
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 502:
                            st.error("Bad Gateway: Nginx cannot reach the prediction service.")
                        elif e.response.status_code == 504:
                            st.error("Gateway Timeout: Request took too long to process.")
                        else:
                            st.error(f"HTTP Error: {e}")

    # Product Prediction
    elif prediction_mode == "Product (Designation + Description)":
        st.markdown("### Product Prediction: `/predict_product`")
        
        designation = st.text_input(
            "Product Designation:",
            placeholder="e.g., Nike Air Max 90"
        )
        
        description = st.text_area(
            "Product Description (optional):",
            placeholder="e.g., Classic running shoes with air cushioning technology",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            return_probs = st.checkbox("Show probabilities", value=True)
        with col2:
            top_k = st.slider("Top predictions", min_value=1, max_value=10, value=5)
        
        if st.button("Predict", type="primary", use_container_width=True):
            if not designation.strip():
                st.warning("Please enter product designation")
            else:
                with st.spinner("Making prediction..."):
                    try:
                        payload = {
                            "designation": designation,
                            "description": description,
                            "return_probabilities": return_probs,
                            "top_k": top_k
                        }
                        
                        response = requests.post(
                            f"{st.session_state.api_url}predict_product",  
                            json=payload,
                            auth=st.session_state.auth,
                            verify=False,
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("Prediction Complete")
                            
                            st.markdown("#### Primary Prediction")
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                st.metric("Category", result.get("predicted_category", "N/A"))
                            with col2:
                                if "confidence" in result:
                                    st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            if "top_k_predictions" in result:
                                st.markdown("#### Top Predictions")
                                
                                for i, pred in enumerate(result["top_predictions"], 1):
                                    class_id = pred.get("category", "N/A")
                                    prob = pred.get("probability", 0)
                                    
                                    st.markdown(f"**{i}. Class {class_id}**")
                                    st.progress(prob)
                                    st.caption(f"Probability: {prob:.4f} ({prob*100:.2f}%)")
                            
                            with st.expander("View Raw API Response"):
                                st.json(result)
                        
                        elif response.status_code == 401:
                            st.error("Authentication failed. Check username and password.")

                        else:
                            st.error(f"API Error: {response.status_code}")
                            st.code(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API through Nginx. Check if services are running.")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The model may be processing a large batch.")
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 502:
                            st.error("Bad Gateway: Nginx cannot reach the prediction service.")
                        elif e.response.status_code == 504:
                            st.error("Gateway Timeout: Request took too long to process.")
                        else:
                            st.error(f"HTTP Error: {e}")

    # Batch Prediction
    else:
        st.markdown("### Batch Prediction: `/predict_batch`")
        
        st.info("Enter one product text per line")
        
        batch_input = st.text_area(
            "Enter multiple products:",
            placeholder="Nike running shoes\nAdidas soccer jersey\nSamsung Galaxy phone",
            height=200
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=32)
        with col2:
            return_probs = st.checkbox("Show probabilities", value=False)
        with col3:
            top_k = st.slider("Top predictions", min_value=1, max_value=10, value=3)
        
        if st.button("Predict Batch", type="primary", use_container_width=True):
            texts = [t.strip() for t in batch_input.split("\n") if t.strip()]
            
            if not texts:
                st.warning("Please enter at least one product text")
            else:
                with st.spinner(f"Processing {len(texts)} predictions..."):
                    try:
                        payload = {
                            "texts": texts,
                            "batch_size": batch_size,
                            "return_probabilities": return_probs,
                            "top_k": top_k
                        }
                        
                        response = requests.post(
                            f"{st.session_state.api_url}predict_batch",  # Updated endpoint
                            json=payload,
                            auth=st.session_state.auth,
                            verify=False,
                            timeout=120
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            predictions = result.get("predictions", [])
                            
                            st.success(f"Processed {result.get('count', 0)} predictions")
                            
                            st.markdown("#### Results")
                            
                            for i, (text, pred) in enumerate(zip(texts, predictions), 1):
                                with st.container():
                                    st.markdown(f"**{i}. {text}**")
                                    col1, col2 = st.columns([1, 3])
                                    with col1:
                                        st.metric("Category", pred.get("predicted_category", "N/A"))
                                    with col2:
                                        if "confidence" in pred:
                                            st.metric("Confidence", f"{pred['confidence']:.2%}")
                                    
                                    if return_probs and "top_predictions" in pred:
                                        with st.expander("View top predictions"):
                                            for j, p in enumerate(pred["top_predictions"], 1):
                                                st.text(f"{j}. Class {p.get('category')}: {p.get('probability', 0):.4f}")
                                    
                                    st.markdown("---")
                            
                            with st.expander("View Raw API Response"):
                                st.json(result)
                        
                        elif response.status_code == 401:
                            st.error("Authentication failed. Check username and password.")

                        else:
                            st.error(f"API Error: {response.status_code}")
                            st.code(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API through Nginx. Check if services are running.")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out. The model may be processing a large batch.")
                    except requests.exceptions.HTTPError as e:
                        if e.response.status_code == 502:
                            st.error("Bad Gateway: Nginx cannot reach the prediction service.")
                        elif e.response.status_code == 504:
                            st.error("Gateway Timeout: Request took too long to process.")
                        else:
                            st.error(f"HTTP Error: {e}")
    # Footer
    st.markdown("---")
    st.caption(f"Connected to: {st.session_state.api_url} | Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Main app logic
if not st.session_state.logged_in:
    login_page()
else:
    prediction_page()
