import streamlit as st
from PIL import Image


# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="ML Ops Setup",
    layout="wide"
)

st.title("ML Ops Setup")

st.write("---")

# ==========GitHub Repository And UV Setup==========
st.subheader("GitHub Repository and UV Setup")
with st.expander("**Problem & Setup**", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Problem:**")
        st.write("""
        * code management, storage and collaboration
        * project reproducibility and package management
        """)

        st.markdown("**Approach:**")
        st.markdown("""
        * git / GitHub workflow: 
          * `main` branch
          * development branches for new features
          * pull requests (`dev` -> `main`) with code reviews
          * GitHub issues for task management
        * `uv` package manager: dependency and virtual environment management
        * uv workspaces:
          * several _workspace members_ managed in single environment
          * each service has own `pyproject.toml` -> `pylock.toml`
        """)

        st.markdown("**Challenges:**")
        st.write("""
        * uv workspaces creates dependencies of python packages between services
        """)

    with col2:
        image_API = Image.open('./streamlit/images/uv.png')
        st.image(image_API, width=1100)

st.write("---")

# ==========Fast API==========
st.subheader("Fast API")
with st.expander("**Problem & Setup**", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Problem:**")
        st.write("""
        Production-ready REST API for model serving and operations
        """)

        st.markdown("**Approach:**")
        st.write("""
        * Microservices: Individual APIs for ML tasks
          * **DATA** preparation
          * model **TRAIN**ing
          * **EVALUATE** trained model
          * model **PREDICT**ion
        * Endpoints, e.g.
          * PREDICT `/predict_text`: simple string input prediction
          * TRAIN `/train_model`: start traning of ML model
          * DATA `/preprocess/from-raw`: raw data preprocessing
          * (ALL) `/health`: check service availability
        * Automatic documentation and validation
        * Serving with `uvicorn`
        """)

        st.markdown("**Challenges:**")
        st.write("""
        * API endpoint development more complex than, e.g., calling python functions
        * Security issues exposing unprotected APIs
        """)

    with col2:
        image_API = Image.open('./streamlit/images/API.png')
        st.image(image_API, width=1100)

st.write("---")

# ==========Docker Containerization==========
st.subheader("Docker Containerization")

# Create expandable sections for each phase
with st.expander("**Problem & Setup**", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Problem:**")
        st.write("""
        Containerization of the different APIs as microservices to
        * disentagles dependencies
        * make services portable, scalable, reproducible
        """)

        st.markdown("**Approach:**")
        st.markdown("""
        
          
        Docker images for each API service:
        * Separate environment setup (build stage) from execution of Docker containers
        * Isolate containers on execution - internet access during build stage only (ideally)
        * Pre-defined images for additional deployments: `nignx`, `prometheus`, `grafana`, ...

        Docker compose:  
        * Orchestrate all services (Docker containers) with dependencies, health checks
        * Manage shared resources: Mounted volumes, networks
        * Manage exposure of services: Port forwarding, networks configuration
        """)

        st.markdown("**Challenges:**")
        st.write("""
        Balance between isolation and sharing of necessary information, exposing interfaces
        """)

    with col2:
        code = """
    # file: docker-compose.yml

    services:
        api_predict:
            build:
                context: src/predict
                dockerfile: Dockerfile
            deploy:
                replicas: 1   # increase for load balancing
            container_name: predict_api
            networks: 
                - api_network
            volumes:
                - ./models:/app/models:ro
                - ./config:/app/config:ro
            healthcheck:
                test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
                interval: 5s
                timeout: 2s
                retries: 12

        nginx:
            build:
                context: deployments/nginx/
                dockerfile: Dockerfile
            container_name: nginx_reverse_proxy
            ports:
                - "8080:8080"
                - "8443:8443"
            networks: 
                - api_network
                - edge_network
                - monitoring_network
            depends_on:
                api_predict:
                    condition: service_healthy

        networks:
            api_network:
                # internal network not exposed to host
                internal: true
            edge_network:
                # network exposed to host for reverse proxy
    """
        st.code(code, language="python")

st.write("---")

# ==========Data And Model Versioning==========
st.subheader("Data And Model Versioning With DVC/Dagshub")
with st.expander("**Problem & Setup**", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Problem:**")
        st.write("""
        * track large datasets and models
        * reproducibility: ensuring use of same data versions
        * team collaboration: sharing of data and models with remote storage
        """)

        st.markdown("**Approach:**")
        st.markdown("""
        DVC for version control: tracking with `.dvc` files
        * raw data
        * preprocessed `.parquet` files
        * model artifacts
        * label encoder files
        * evaluation results
                    
        DagsHub as remote storage: 
        * centralized cloud storage for team access  
        * lightweight files to GitHub: Only metadata (`.dvc` files) in git
        """)

        st.markdown("**Challenges:**")
        st.write("""
        * Powerful platform with many features (data, experiments, models, ...)
        * Management of user access tokens
        """)

    with col2:
        #image_GF = Image.open('./streamlit/images/Dagshub.png')
        image_GF = Image.open('./streamlit/images/dagshub_dvc.png')
        st.image(image_GF, width=1100)

st.write("---")

# ==========Experiment Tracking With MLFlow==========
st.subheader("Experiment Tracking With MLFlow")
with st.expander("**Problem & Setup**", expanded=False):
    st.markdown("**Problem:**")
    st.write("""
    * compare model performance across different parameters and model setups
    * track metrics, metadata 
    * enable decision making for model selection
    """)

    st.markdown("**Approach:**")
    st.write("""
    * local MLFlow server
    * logging of metadata and model metrics
    * persistent storage with mounted mlflow volume
    """)

    st.markdown("**Challenges & Outlook:**")
    st.write("""
    * logging of model artifacts
    * model registry and serving
    * switch to Dagshub MLFlow for cloud-based tracking can slow down training
    * add evaluation results to tracking
    """)

    with st.expander("**MLFlow UI**", expanded=False):
        image_MF = Image.open('./streamlit/images/MLFlow_Overview.png')
        st.image(image_MF, width=1100)

        image_MF2 = Image.open('./streamlit/images/MLFlow_Run_Example_Overview.png')
        st.image(image_MF2, width=1100)

        image_MF3 = Image.open('./streamlit/images/MLFlow_Run_Example_Metrics.png')
        st.image(image_MF3, width=1100)

st.write("---")

# ==========Reverse Proxy With Nginx==========
st.subheader("Reverse Proxy With Nginx")
with st.expander("**Problem & Setup**", expanded=False):
    st.markdown("**Problem:**")
    st.write("""
    Unique entry point to all functionality, only single (web) address
    """)

    st.markdown("**Approach:**")
    st.markdown("""
    Forwards requests to responsible API services, e.g.  
    `https://nginx:8443/predict/predict_text` -> `http://predict_api:8000/predict_text`
    
    Security features:  
    * First gatekeeper: Manages authentication
      * implemented: HTTP basic with username+password on each request
      * User roles (_user_, _developer_, _admin_) have access to different APIs
    * Encrypted communication: Only HTTPS to "external world", redirect HTTP requests
    * Allow/block access to endpoints (`/nginx_status` only on monitoring network)
    * Limit API requests, eg. 20 requests / second (counter DoS attacks)
    * Allows load balancing between several backends (prepared, but currently only one instance of each service)  
    """)

    st.markdown("**Challenges:**")
    st.write("""
    * Forwarding of automatically generated API docs, web interfaces of prometheus, grafana, e.g. `predict_api:8000/docs` as `nginx:8443/predict/docs`
    * Generate (self-signed) SSL certificates accepted for different addresses (`nginx`, `localhost`) and tools (curl, python requests, web browsers).
    """)

st.write("---")

# ==========CI/CD==========
st.subheader("Automated Testing & CI")
with st.expander("**Problem & Setup**", expanded=False):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Problem:**")
        st.write("""
        Discover disfunctionalities during development as soon as possible by automated testing
        """)

        st.markdown("**Approach:**")
        st.write("""
        CI workflow with GitHub Actions
        * Code linting and formatting checks with ruff
        * Triggered on each push and pull requests (`main`)
                 
        API tests on system startup
        * Implemented with pytest
        * Running from individual Docker containers
        * On each startup of full system (`docker compose up`)
        * Test requests to API containers, verify response (positive+negative)
        * Logs written to stdout (`docker compose logs`) and log files (outside of container)
        """)

        st.markdown("**Challenges:**")
        st.write("""
        * "Realistic setting": From where should test requests originate? (host system, edge_network, ...)
        * Balance between test coverage and execution speed
        * Test results: Visibility, readability (+ ideally automated processing)
        """)

    with col2:
        code = """
        ============================= test session starts ==============================
        platform linux -- Python 3.11.14, pytest-9.0.2, pluggy-1.6.0
        Session start: 2026-02-11 11:21:31+0000
        rootdir: /tests
        configfile: pyproject.toml
        collected 37 items

        test_api_predict.py .....................................                [100%]

        ============================== 37 passed in 0.51s ==============================
        """
        st.code(code, language="python")
st.write("---")


# ==========Performance Monitoring With Prometheus/Grafana==========
st.subheader("Monitoring With Prometheus/Grafana")
with st.expander("**Problem & Setup**", expanded=False):
    st.markdown("**Problem:**")
    st.markdown("""
    Provide accessible way / overview of the system performance (+ ideally model performance)

    -> Early indication of bottlenecks, performance issues and help to track down errors.
    """)

    st.markdown("**Approach:**")
    st.markdown("""
    * APIs provide performance metrics on `/metrics` endpoints with `prometheus_fastapi_instrumentator`
    * nginx provides status on `/nginx_status` endpoint  
      * `nginx-prometheus-exporter` transforms to Prometheus readable format
    * Prometheus scrapes metrics endpoints, exporters and stores time series  
    * Grafana visualizes the data from Prometheus in dashboards  
      * Provisioning: Dashboards and data sources persistent as JSON files
    """)

    st.markdown("**Challenges:**")
    st.markdown("""
    * Involves mighty tools with lots of features (rely on predefined solutions)  
    * Make Grafana dashboards persistent (without storing full database): Provisioning
    """)

    with st.expander("**Dashboards**", expanded=False):
        image_GF = Image.open('./streamlit/images/Grafana_1.png')
        st.image(image_GF, width=800)

        image_GF2 = Image.open('./streamlit/images/Grafana_2.png')
        st.image(image_GF2, width=800)
st.write("---")