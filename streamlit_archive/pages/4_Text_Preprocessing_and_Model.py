# imports
import streamlit as st
import pandas as pd
from PIL import Image

# Simple page config
st.set_page_config(
    page_title="Text Preprocessing & Modeling",
    page_icon="📝",
    layout="wide"
)

# st.warning("⚠️ **Work in Progress** - This page is still under active development")
st.write("")
st.write("Date: 1st December 2025")
st.write("")

# ============================================================================
# HEADER
# ============================================================================

st.title("📝 Text Preprocessing & Modeling")

st.write("""
Comprehensive text preprocessing pipeline and baseline model development for the Rakuten classification challenge.
""")

st.write("---")

# ============================================================================
# SECTION 1: TEXT CLEANING & NORMALIZATION (Slide 1)
# ============================================================================

st.subheader("🧹 Text Cleaning & Normalization")

col1, col2, col3 = st.columns(3)

with col1:
    # Image from slide 1
    try:
        slide1_img1 = Image.open('./images/ppt_images/slide_1_image_5_1.png')
        st.image(slide1_img1, use_container_width=True)
    except:
        st.info("💡 Place slide_1_image_5.png in ./images/ppt_images/ folder")
    st.write("""
    **📄 Raw Text Sources**
    - Webpages
    - Databases
    - User-generated text
    
    **Structural & Artifact cleaning:**
    - Remove HTML / XML entities
    - Fix malformed text
    - Remove special characters / emojis
    """)

with col2:
    # Image from slide 1
    try:
        slide1_img2 = Image.open('./images/ppt_images/slide_1_image_5_2.png')
        st.image(slide1_img2, use_container_width=True)
    except:
        st.info("💡 Place slide_1_image_5.png in ./images/ppt_images/ folder")
    st.write("""
    **🔧 Encoding Error Correction**
    - "MATELAS:Â·" → "MATELAS:"
    - "Massante\\"Â" → "Massante"
    - "cm.Â" → "cm"
    
    **Redundancy & Outliers:**
    - Sentence Outlier Detection
    - NLTK sentence tokenizer
    - Dynamic threshold counter
    - Sentence Level Redundancy removal
    - TF-IDF vectorization (scikit-learn)
    - Cosine similarity (duplicate remove)
    - Frequency based summarization
    - Spacy + en_core_web_sm
    - Select most informative sentences
    """)

with col3:
    # Image from slide 1
    try:
        slide1_img3 = Image.open('./images/ppt_images/slide_1_image_5_3.png')
        st.image(slide1_img3, width=163)
    except:
        st.info("💡 Place slide_1_image_5.png in ./images/ppt_images/ folder")
    st.write("""
    **✅ Outcome**
    - Cleaned Text
    - Drop Duplicate
    - Removed Artifacts
    - Summarized text
    
    **Advantage:**
    - Improve tokenization & Embeddings
    """)

st.write("---")

# ============================================================================
# SECTION 2: TEXT STANDARDIZATION (Slides 2, 3, 4)
# ============================================================================

st.subheader("🌐 Text Standardization")

# Slide 2 content
st.markdown("### Challenge")
st.write("""
**Challenge:** Highly unstructured multilingual dataset (~27 languages) to unified English translated.

**Tools:** Local LLMs or Google, DeepL etc. (API)

**Shortcomings:** Local LLMS (row-wise translation) too slow, Others - faster but paid version
""")

st.write("")

# Slide 3 - Multi-lingual text diagram
st.markdown("### Diagram")
st.write("")

# Note: Slide 3 has diagram shapes but no extractable image - add placeholder
st.info("💡 **Diagram:** Multi-lingual text → Translation → Unified English text")
try:
    slide1_img = Image.open('./images/ppt_images/slide_3_image_5.png')
    st.image(slide1_img, use_container_width=True)
except:
    st.info("💡 Place slide_1_image_5.png in ./ppt_images/ folder")

st.write("")

# Slide 4 content
st.markdown("### Post-Translation Clean-up")
st.write("""
**Post-Translation Artifacts:**
- Remaining noise, characters, symbols, glued compound tokens (e.g., 500mlCapacity, IP67Waterproof), and inconsistent punctuation.

**Comprehensive clean up:**
- Removed Emails, excess white space, space between letter-number (500ml → 500 ml)
- Word segment (e.g. WaterResistance → Water Resistance)
""")

st.write("---")

# ============================================================================
# SECTION 3: PREPROCESSING FOR MODEL LEARNING (Slide 5)
# ============================================================================

st.subheader("⚙️ Data Preparation")

# Slide 5 diagram as text flow

col1, col2 = st.columns(2)

with col1:

    st.write("""
    **Some Further preprocessing steps for model learning:**
    - Label Encoding
    - Stratified Split
    - Class weight
    - Tokenizer
    """)

with col2:
    try:
        slide1_img = Image.open('./images/ppt_images/slide_5_image_5.png')
        st.image(slide1_img, width=300)
    except:
        st.info("💡 Place slide_1_image_5.png in ./ppt_images/ folder")

st.write("---")

# ============================================================================
# SECTION 4: BASELINE MODELS - LSTM (Slides 6, 7, 8)
# ============================================================================

st.subheader("🤖 Baseline Models – LSTM, BiLSTM, GRU")

st.markdown("### LSTM")

# Placeholder text
st.write("""

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) capable of learning 
long-term dependencies by regulating and learning the information using different gates.
""")

st.write("")

# Slide 6 - LSTM architecture diagram
st.markdown("#### Architecture")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**LSTM Architecture:**")
    st.code("""
Input Sequence
(batch, 200)
    ↓
Spatial Dropout 1D (0.2)
    ↓
Embedding
60k * 128 (trainable)
7.68M Params
    ↓
LSTM
128 Units
    ↓
Dropout (0.3)
    ↓
Dense (ReLU)
128
    ↓
Dropout (0.3)
    ↓
Dense (Softmax)
27 Classes
    """)

with col2:
    try:
        slide6_img = Image.open('./images/ppt_images/slide_6_image_5_1.png')
        st.image(slide6_img, use_container_width=True)
        slide62_img = Image.open('./images/ppt_images/slide_6_image_5_2.png')
        st.image(slide62_img, use_container_width=True)
    except:
        st.info("💡 Place image.png in ./ppt_images/ folder")

st.write("")

# Slide 7 content

# Model Setup
st.markdown("### Model Setup")

st.write("")

st.markdown("**Training Settings:**")
st.write("""
- Loss: sparse_categorical_crossentropy
- Optimizer: Adam(lr=0.001)
- Class_weight = 'balanced'
- EarlyStopping
""")

st.markdown("**Analysis Outcome:**")
try:
    slide7_img = Image.open('./images/ppt_images/slide_7_image_5.png')
    st.image(slide7_img, use_container_width=True)
except:
    st.info("💡 Place image.png in ./ppt_images/ folder")

st.write("")

# Slide 8 - Results
st.markdown("### Results")

st.write("""
**Overall performance:** 75% accuracy & 0.75 weighted F1-score – baseline for 27 imbalanced classes

**Strong classes** (e.g., 5, 9, 14, 15): high precision/recall → clear and distinctive textual patterns

**Challenging classes** (e.g., 0, 6, 8): lower recall → overlapping descriptions and rarity hurt performance
""")

st.write("")

# Images from slide 8
col1, col2 = st.columns(2)

with col1:
    try:
        slide8_img1 = Image.open('./images/ppt_images/slide_8_image_5_1.png')
        st.image(slide8_img1, caption="Confusion Matrix", use_container_width=False)
    except:
        st.info("💡 Place image .png in ./ppt_images/ folder")

with col2:
    try:
        slide8_img2 = Image.open('./images/ppt_images/slide_8_image_5_2.png')
        st.image(slide8_img2, caption="Performance Metrics", use_container_width=True)
    except:
        st.info("💡 Place image .png in ./ppt_images/ folder")

st.write("---")

# ============================================================================
# SECTION 5: BI-LSTM (Slides 9, 10, 11)
# ============================================================================

st.markdown("### Bi-LSTM")

# Placeholder text
st.write("""
Bidirectional LSTM (Bi-LSTM) processes text sequences in both forward and backward directions, 
capturing context from both past and future tokens. This bidirectional approach often improves 
performance on text classification tasks.
""")

st.write("")

st.markdown("### Model Setup")

# Slide 9 - Bi-LSTM architecture diagram
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Bi-LSTM Architecture:**")
    st.code("""
Input Sequence
(batch, 200)
    ↓
Embedding
60k * 300 (trainable)
    ↓
Spatial Dropout 1D (0.2)
    ↓
Bi-LSTM
128 Units
    ↓
Bi-LSTM
64 Units
    ↓
Dropout (0.4)
    ↓
Dense (ReLU)
256
    ↓
Dropout (0.4)
    ↓
Dense (Softmax)
27 Classes
    """)

with col2:
    st.write("")
    st.write("")
    st.write("")
    try:
        slide9_img = Image.open('./images/ppt_images/slide_9_image_5.png')
        st.image(slide9_img, caption="Performance Metrics", use_container_width=True)
    except:
        st.info("💡 Place image .png in ./ppt_images/ folder")

st.write("")

# Slide 10 content

st.markdown("**Training Settings:**")
st.write("""
- Loss: sparse_categorical_crossentropy
- Optimizer: Adam(lr=0.001)
- Class_weight = 'balanced'
- EarlyStopping, ReduceLROnPlateau
""")

st.markdown("**Analysitcome:**")
try:
    slide10_img = Image.open('./images/ppt_images/slide_10_image_5.png')
    st.image(slide10_img, use_container_width=True)
except:
    st.info("💡 Place image.png in ./ppt_images/ folder")

st.write("")

# Slide 11 - Bi-LSTM Results
st.markdown("### Results")

st.write("""
**Overall performance:** ~72.9% validation accuracy & 0.73 weighted F1-score – strong result for highly imbalanced 27-class problem

**Strong classes** (e.g., 3, 5, 9, 13, 14): high precision/recall → distinctive and frequent textual patterns well captured by Bi-LSTM

**Challenging classes** (e.g., 0, 1, 6, 8): lower recall/F1 → rare or ambiguous categories remain difficult despite class weighting
""")

st.write("")

# Images from slide 11
col1, col2 = st.columns(2)

with col1:
    try:
        slide11_img1 = Image.open('./images/ppt_images/slide_11_image_5_1.png')
        st.image(slide11_img1, caption="Confusion Matrix", use_container_width=True)
    except:
        st.info("💡 Place slide_11_image_1.png in ./ppt_images/ folder")

with col2:
    try:
        slide11_img2 = Image.open('./images/ppt_images/slide_11_image_5_2.png')
        st.image(slide11_img2, caption="Performance Metrics", use_container_width=True)
    except:
        st.info("💡 Place image .png in ./ppt_images/ folder")

st.write("---")

# ============================================================================
# SECTION 6: GRU (Slide 12)
# ============================================================================

st.markdown("### GRU")

# Placeholder text
st.write("""
Gated Recurrent Unit (GRU) is a simplified variant of LSTM that uses fewer parameters while 
maintaining comparable performance. GRUs are computationally more efficient and often perform 
well on text classification tasks.
""")

st.write("")

st.markdown("### Results")

st.write("""
**Overall performance:** ~76.5% validation accuracy – best result so far, clear improvement over LSTM/Bi-LSTM

GRU's simpler gating mechanism (vs. LSTM/Bi-LSTM) appears more effective for this task – faster convergence, better generalization
""")

st.write("")

# Images from slide 12
col1, col2 = st.columns(2)

with col1:
    try:
        slide12_img1 = Image.open('./images/ppt_images/slide_12_image_5_1.png')
        st.image(slide12_img1, caption="Confusion Matrix", use_container_width=True)
    except:
        st.info("💡 Place slide_12_image_2.png in ./ppt_images/ folder")

with col2:
    try:
        slide12_img2 = Image.open('./images/ppt_images/slide_12_image_5_2.png')
        st.image(slide12_img2, caption="Performance Metrics", use_container_width=True)
    except:
        st.info("💡 Place slide_12_image_3.png in ./ppt_images/ folder")

st.write("---")

# ============================================================================
# SECTION 7: HYPERPARAMETER STUDY (Slide 13)
# ============================================================================

st.subheader("🔬 Hyperparameter Study")

st.write("""
Search conducted using **Keras Tuner** with **Bayesian Optimization**
""")

st.markdown("### Parameters Taken Into Consideration")

# Create table for hyperparameters
col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Architecture Parameters:**
    - Embedding dimension: {128, 200, 256, 300}
    - RNN type: {LSTM, Bidirectional LSTM, GRU}
    - RNN units: 128–512 (step 32)
    - Dense layer size after RNN: 96–256
    """)

with col2:
    st.write("""
    **Regularization & Optimization:**
    - Spatial dropout after embedding: 0.20–0.50
    - RNN and dense-layer dropout rates: 0.20–0.70
    - Learning rate (log scale): 8×10⁻⁴ – 2.5×10⁻³
    """)

try:
    slide13_img = Image.open('./images/ppt_images/slide_13_image_5_1.png')
    st.image(slide13_img, caption="Hyperparameter Study Results", width=200)
except:
    st.info("💡 Place image .png in ./ppt_images/ folder")

st.write("")

# Hyperparameter study results image from slide 13
try:
    slide13_img = Image.open('./images/ppt_images/slide_13_image_5_2.png')
    st.image(slide13_img, caption="Hyperparameter Study Results", use_container_width=True)
except:
    st.info("💡 Place image .png in ./ppt_images/ folder")

st.write("---")

# ============================================================================
# SECTION 8: TRAIN AND VALIDATION LOSS (Slide 14)
# ============================================================================

st.subheader("📉 Train and Validation Loss")

try:
    slide14_img = Image.open('./images/ppt_images/slide_14_image_5.png')
    st.image(slide14_img, use_container_width=True)
except:
    st.info("💡 Place image .png in ./ppt_images/ folder")

st.write("---")

# ============================================================================
# SECTION 9: ACCURACY (Slide 15)
# ============================================================================

st.subheader("🎯 Accuracy")

try:
    slide15_img = Image.open('./images/ppt_images/slide_15_image_5.png')
    st.image(slide15_img, use_container_width=True)
except:
    st.info("💡 Place image .png in ./ppt_images/ folder")