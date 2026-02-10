# imports
import streamlit as st
import pandas as pd
from PIL import Image

# --- Functions ---

# Image function

def show_photo(photo_name, target_height=250):
    """Display photo with correct height & aspect ratio - no caching needed"""
    
    try:
        img = Image.open(f"./photos/{photo_name}.jpg")
        ratio = img.width / img.height
        img_resized = img.resize((int(target_height * ratio), target_height))
        st.image(img_resized)
    except FileNotFoundError:
        st.error(f"Photo not found!")

# --- Streamlit Page ---

# Simple page config
st.set_page_config(
    page_title="Data Scientest Project Defense",
    page_icon="🛍️",
    layout="wide"
)

# st.warning("⚠️ **Work in Progress** - This page is still under active development")
st.write("")
st.write("Date: 1st December 2025")
st.write("")

# Main title
st.title("🛍️Data Scientest Project Defense!")
st.subheader("Rakuten E-commerce Product Classification Challenge")

st.write("---")

st.title("📊 Project Overview")

# ========== THE PROBLEM ==========
st.subheader("🎯 Project Description")
st.write("This Project is based on the Rakuten France dataset and aims to classify e-commerce products into 27 categories using both text descriptions and product images.")
st.write("The dataset from Rakuten France contains 84,916 products with multilingual text descriptions (French, German, English, etc.) with associated images.")
st.write("Main Goal is to build a multi-modal deep learning model that effectively combines text and image data to accurately classify products into their respective categories.")
st.write("See the following URL for more details: https://challengedata.ens.fr/challenges/35")
st.write("")
st.write("This DataScientest project was rated on difficulty level 10/10.")

st.write("---")

#st.markdown('<p style="font-size:18px">Custom Size Text</p>', unsafe_allow_html=True)

# Key metrics
st.subheader("📊 Project Key Facts")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Products", "84,916")
col2.metric("Categories", "27")
col3.metric("Images", "3GB")
col4.metric("Languages", "30+")

st.write("---")

# ========== OUR GOAL ==========
st.subheader("🎯 Our Goal")

col1, col2, = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #8c564b; padding: 20px; border-radius: 10px; text-align: center;">
        <p style="margin: 0; font-size: 20px; color: white;">Method</p>
        <h2 style="margin: 10px 0; font-size: 36px; color: white;">Multimodal Classification</h2>
        <p style="margin: 0; font-size: 14px; color: white;">pretrained image & text model</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #d62728; padding: 20px; border-radius: 10px; text-align: center;">
        <p style="margin: 0; font-size: 20px; color: white;">Target Accuracy</p>
        <h5 style="margin: 10px 0; font-size: 48px; color: white;">> 80%</h5>
        <p style="margin: 0; font-size: 14px; color: white;">minimum</p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# Our Approach - Simple tabs
st.subheader("🚀 Our Project Approach")

tab1, tab2, tab3, tab4 = st.tabs(["Text", "Images", "Fusion", "Orga"])

with tab1:
    st.write("**Text Pipeline:**")
    st.write("1. Clean data (remove HTML, special chars, etc)")
    st.write("2. Translate to English (via local models, Google/LLM)")
    st.write("3. Balance dataset (undersampling)")
    st.write("4. Research text models for next stage (LSTM, BERT, etc.)")

with tab2:
    st.write("**Image Pipeline:**")
    st.write("1. Remove blanks, duplicates, and noisy images")
    st.write("2. Crop backgrounds")
    st.write("3. Data augmentation (flip, rotate, zoom)")
    st.write("4. Exploration of Pretrained Image Models")

with tab3:
    st.write("**Fusion Strategy:**")
    st.write("1. Train text model → get features")
    st.write("2. Train image model → get features")
    st.write("3. Combine features")
    st.write("4. Final classifier → 27 categories")

with tab4:
    st.write("**Organization:**")
    st.write("1. Reporting")
    st.write("2. Overall Research & Exploration")
    st.write("3. Computing Infrastructure Setup")
    st.write("4. Presentation")

st.write("---")

# --- Porject Workflow ---
st.subheader("🔄 Project Pipeline Workflow & Tasks")

st.write("""
Our project follows a systematic approach with four main phases.
""")

# Create expandable sections for each phase
with st.expander("**1️⃣ Data Analysis & Exploration**", expanded=False):
    st.markdown("**Data Analysis:**")
    st.write("""
    * Statistical Exploration (Image and Text)
    * Class Distribution
    * Missing Values
    * Duplicates (Text and Images)
    * Word and Sentence Frequencies
    * Languages
    * Images Size and Quality
    * Blank and Noisy Images
    * Background
    * Pixel Distributions
    """)

with st.expander("**2️⃣ Data Preprocessing**", expanded=False):
    
    # Use tabs for text vs image preprocessing
    tab1, tab2, tab3 = st.tabs(["📝 Text", "🖼️ Image", "⚙️ Pipeline"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Text Preprocessing")
            st.write("""
            **HTML & Format Cleanup:**
            - Remove HTML tags and convert to natural language
            - Remove special characters, URLs, emails
            - Handle spelling corrections to closest valid words
            - Separate joined words (e.g., mmBottle → mm Bottle)
            - Lowercase all textual data
            - Remove products with missing descriptions
            - Remove Duplicates and Blanks (Images, Text)
            
            **Content Optimization:**
            - Remove descriptions duplicated in designation
            - Merge designation and description columns
            - Remove repetitions in text
            - Remove redundant Information (Similar Sentences, Text Summary)
            - Summarize overly long descriptions
            
            **Data Quality:**
            - Define and remove outliers (too short/long text)
            - Keep natural language (preserve stop words)
            - Words encoding (Handle spelling correction to most closest words)                 
            - Balance Dataset (several sampling techniques)
            - Splitting Train Dataset (no Y_test.csv available, prevent Data Leakage)
            """)
        with col2:
            st.markdown("##### Translation & Standardization")
            st.write("""
            **Language Unification:**
            - Detected 30+ languages (60% French, 20% German, 20% others)
            - Target language: English
            - Translation of whole dataframe using:
            - Google Translate API
            - DeepL API
            - Local LLM (llama model)
            - Final choice: BERT model (best speed/quality trade-off)
            - Full dataset translation: 50K products in ~2 hours
            """)    
    
    with tab2:
        st.markdown("##### Image Preprocessing")
        st.write("""
        **Image Cleaning:**
        - Crop white/blank margins (zoom to reduce whitespace)
        - Define pixel thresholds for margin detection
        - Check and remove duplicate images
        - Remove noise and artifacts
        - Identify and remove blank images (no information)
        
        **Standardization:**
        - Resize to standard dimensions (500×500)
        - Normalize pixel distributions
        - Optional: Convert to grayscale (tested, not used in final)
        
        **Data Augmentation:**
        - ImageDataGenerator for training
        - Transformations: zoom, rotate, flip, shift
        - Applied during training stage (not preprocessing)
        - Increases variety and reduces overfitting
                 
        **Images**
        - Lower white margin of Images
        """)
    
    with tab3:
        st.markdown("##### Dataset Preparation")
        st.write("""
        **Class Imbalance Strategy:**
        - Hybrid approach evaluated:
          - Option A: Delete some elements (based on quality/random)
          - Option B: Oversample using ImageDataGenerator
          - Option C: Use class weights (final choice)
        - Final decision: Class weights during training
        - Removed lowest quality samples (40.8% of data)
        
        **Dataset Splitting:**
        - Train / Validation / Test split
        - Prevent data leakage considerations
        - Stratified sampling to maintain class distribution
        
        **Encoding & Embeddings:**
        - Label encoding for 27 categories
        - Optional: Use class names instead of codes
        - Text embeddings preparation (BERT tokenization)
        - Image feature extraction (CNN layers)
        
        **Infrastructure:**
        - Automated preprocessing pipeline
        - Modular design for reproducibility
        - Version control for datasets
        - Cloud computing setup (Colab, AWS)

        **Pipeline**
        - Label Encoding
        - Embeddings
        - Hyper-parameter study using all above models (Basic LSTM, Bidirectional LSTM, GRU)
        """)

with st.expander("**3️⃣ Modeling & Training**", expanded=False):
    
    # Use tabs for different model types
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Models", "🖼️ Image Models", "🔗 Fusion", "⚙️ Infrastructure"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Basic Text Models (Baseline)")
            st.write("""
            **Architecture Experiments:**
            - LSTM (Long Short-Term Memory)
            - Bidirectional LSTM
            - GRU (Gated Recurrent Unit)
            
            **Feature Extraction:**
            - TF-IDF vectorization
            - Word2Vec embeddings
            - Character n-grams (2-5 grams)
            
            **Hyperparameter Tuning:**
            - Learning rate optimization
            - Batch size experiments
            - Hidden layer dimensions
            - Dropout rates
            """)
        
        with col2:
            st.markdown("##### Transfer Learning (Text)")
            st.write("""
            **Model Selection:**
            - BERT (Bidirectional Encoder Representations from Transformers)
            
            **Data Adaptation:**
            - BERT tokenization (WordPiece)
            - Maximum sequence length: 512 tokens
            - Padding and truncation strategies
            - Special tokens: [CLS], [SEP]
            
            **Training Strategy:**
            - Fine-tuning pre-trained BERT
            - Freeze early layers, train later layers
            - Learning rate: 2e-5
            - Class weights for imbalance
            
            **Challenges:**
            - Computational resource limitations
            - Training time: ~6 hours on T4 GPU
            - Memory constraints with batch size
        """)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Basic Image Models (Baseline)")
            st.write("""
            **Vanilla CNN Architecture:**
            - Multiple convolutional layers
            - MaxPooling for dimension reduction
            - Fully connected layers
            - Softmax output (27 classes)
            
            **Training Details:**
            - Data augmentation applied
            - Batch normalization
            - Dropout for regularization
            
            **Results:**
            - Current accuracy: ~55%
            - Significant room for improvement
            """)
            
        #st.markdown("<br>", unsafe_allow_html=True)  # ← Use this instead
        with col2:
            st.markdown("##### Transfer Learning (Image)")
            st.write("""
            **Model Selection:**
            - Primary candidates: ResNet, EfficientNet, MobileNet
            - Trade-off: Accuracy vs computational cost
            
            **Pre-trained Models:**
            - ImageNet pre-training
            - Fine-tuning on Rakuten data
            - Feature extraction from deep layers
            
            **Hardware Solutions:**
            - TensorFlow Metal for M1/M2 Macs
            - Kaggle GPU resources (T4, P100)
            - Google Colab Pro
            - AWS EC2 instances (optional)
            
            **Optimization:**
            - Learning rate scheduling
            - Early stopping
            - Model checkpointing
            """)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.write("""
            **Transfer Learning (Text)**
            - Model Selection: BERT

            **Transfer Learning (Image)**
            - Model Selection: EfficientNetB0, ConvNextTiny, Basic CNN
            
            **Fusion Strategy:**
            - Combine text features + image features
            - Concatenation vs attention-based fusion
            - Investigating PyTorch implementations
            
            **Architecture:**
            - Separate feature extraction branches
            - Fusion layer (concatenate embeddings)
            - Shared classification head
            - Joint training vs separate training
            """)
        with col2:
            st.write("""
            **Research:**
            - Literature review on multimodal learning
            - Best practices for text-image fusion
            - Loss function design for combined models
            
            **Expected Improvement:**
            - Text alone: ~75%
            - Image alone: ~50-60%
            - **Fusion target: 80-90%**
            """)
    
    with tab4:
        st.markdown("##### Cloud Computing & Infrastructure")
        st.write("""
        **Platforms Evaluated:**
        - ⚙️ Google Colab (free T4 GPU, used for translation)
        - ⚙️ Kaggle Kernels (free P100 GPU, 30h/week)
        - ⚙️ GitHub Codespaces (considered)
        - ⚙️ AWS EC2 (investigated, optional)
        
        **Local Solutions:**
        - TensorFlow Metal for local M1/M2 training
        - GPU memory optimization techniques
        - Batch size adjustment for limited VRAM
        
        **Collaboration:**
        - GitHub for version control
        - Google Docs for reporting
        - Shared datasets via Google Drive & Slack
        - Regular sync meetings
        """)

with st.expander("**4️⃣ Results & Deliverables**", expanded=False):

    st.markdown("##### Project Deliverables")
    st.write("""
    **Technical Outputs:**
    - Preprocessing & modeling pipeline code
    - Final Fusion Model
    
    **Documentation:**
    - Full technical report (Google Docs)
    - Code documentation via Github
    - Final interactive Streamlit defence presentation (see repo)
    """)

st.write("---")

st.subheader("🔄 Our Methodology")

methodology = "./images/methodology.png"
st.image(methodology, caption="Our Methodology Overview", use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Data Pipeline**")
    st.write("""
    - Data Exploration
    - Text & Image Preprocessing
    - Quality Control
    """)

with col2:
    st.markdown("**Uni-modal Models**")
    st.write("""
    - BERT (Text)
    - Custom CNN (Images)
    - EfficientNet (Transfer Learning)
    """)

with col3:
    st.markdown("**🔗 Multi-modal Fusion**")
    st.write("""
    - Late Fusion Architecture
    - Combined Features
    - Final Classification
    """)

st.write("---")

# Key Findings
st.subheader("💡 Key Discoveries")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("**What Worked**")
    st.write("- Quick text cleaning")
    st.write("- Local model translation")
    st.write("- Image Data Augmentation")
    st.write("- Class weights in transfer learning")

with col2:
    st.error("**What Didn't Work**")
    st.write("- Computing limitations with translation & image training")
    st.write("- Balancing via undersampling")

with col3:
    st.info("**Surprises**")
    st.write("- Text quality > balance")
    st.write("- Image data less impact")
    st.write("- Model ensembling worked out better than expected")

st.write("---")

# Current Status
st.subheader("📈 Current Status")

col1, col2 = st.columns([2, 1])

with col1:
    status_data = pd.DataFrame({
        'Phase': ['Data Exploration', 'Text Preprocessing', 'Image Preprocessing', 
                  'Text Model', 'Image Model', 'Fusion', 'Deployment'],
        'Status': ['✅ Complete', '✅ Complete', '✅ Complete',
                   '✅ Complete', '✅ Complete', '✅ Complete', '✅ Complete' ]
    })
    #'🔄 Training', '🔄 Training', '⏳ Pending']
    st.dataframe(status_data, hide_index=True, use_container_width=True)

with col2:
    st.metric("Baseline", "~76.5%")
    st.metric("Current", "85.0%", delta="+8.5%")
    st.metric("Target Accurary", "80-90%")

st.write("---")