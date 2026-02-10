#imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

# Simple page config
st.set_page_config(page_title="Image Preprocessing & Modeling", page_icon="🖼️", layout="wide")

# st.warning("⚠️ **Work in Progress** - This page is still under active development")

st.title("🖼️ Image Preprocessing & Modeling")

# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Image preprocessing focus**

    1. **Quality Filtering:** Remove blank images and duplicates (SSIM-based)
    2. **Background Detection:** Classify white vs. colored backgrounds
    3. **Standardization:** 500×500 pixels, organized by category
    """)

with col2:
    st.write("""
    **Image modeling focus**

    1. **Baseline Model** Basic CNN
    2. **Transfer Learning Model** EfficientNet
    """)

st.write("---")

# ============================================================================
# IMAGE STATISTICS
# ============================================================================

st.subheader("📊 Image Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Images",
        value="84,916"
    )

with col2:
    st.metric(
        label="Dimensions",
        value="500×500px",
        help="All images standardized"
    )

with col3:
    st.metric(
        label="Blank Images",
        value="0.01%",
        delta="7 images"
    )

with col4:
    st.metric(
        label="White Background",
        value="89.2%",
        help="Images with >20% white pixels"
    )

st.write("---")

# ============================================================================
# SECTION 1: QUALITY FILTERING
# ============================================================================

st.subheader("1️⃣ Quality Filtering")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔍 Blank Image Detection**")
    st.metric("Blank Images", "7 (0.01%)")
    
    st.write("""
    **Method:**
    - Match variance = 0 to find blank images
    """)

with col2:
    st.markdown("**🔄 Duplicate Detection**")
    st.write("""
    **Method:** SSIM-based (Structural Similarity Index)
    - 3873 duplicates found & removed
    - Threshold: 0.95
    - Two-stage approach:
      1. Group by identical mean grayscale
      2. Compute SSIM within groups
    
    **Advantage:** Avoids exhaustive pairwise comparisons
    """)

st.write("---")

# ============================================================================
# SECTION 2: BACKGROUND DETECTION
# ============================================================================

st.subheader("Image Preprocessing")

st.subheader("2️⃣ Background Detection Strategy")


st.markdown("**⬜ White Background Detection**")
st.write("""
**Method:** Border region analysis
- Analyze 10% of border pixels
- Threshold: Pixel intensity > 240
- Classify as white if ≥90% of border pixels are white

**Result:** 89.2% of images have white backgrounds
""")
st.info("💡 Typical for e-commerce photography (clean white backgrounds)")

st.write("---")

# ============================================================================
# SECTION 3: IMAGE MODEL TRAINING
# ============================================================================

st.subheader("3️⃣ Image Model Training")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🛍️ Basic CNN**")
    
    image_cnn = Image.open('./images/basic_cnn_architecture.png')
    st.image(image_cnn, caption='CNN Model Architecture', use_container_width=True)
    
    st.markdown("""
        **Data Augmentation:**
        - Geometric transformations:
        - Random horizontal flip (p=0.5)
        - Random vertical flip (p=0.3)
        - Random rotation (30°)
        - Random resized crop (scale=0.8-1.0)         
    """)
    st.markdown("""
        **Color Augmentation:**
        - ColorJitter (brightness, contrast, saturation, hue ±0.3/0.1)
        - Random grayscale conversion (p=0.1)
        - Gaussian blur (p=0.3, sigma=0.1-2.0)         
    """)
    st.markdown("""
        **Regularization augmentation:**
        - Random erasing (p=0.25, scale=0.02-0.2):
        Randomly masks rectangular regions to improve robustness to occlusion
        """)

with col2:
    st.markdown("**🛍️ EfficientNet**")

    image_effinet = Image.open('./images/efficientnet_architecture_simple.png')
    st.image(image_effinet, caption='EfficientNet Model Architecture', use_container_width=True)

    st.write("""
    **EfficientNet only**
    - Weighted Sampling
    - Class weighted loss
    """)

st.write("---")

st.subheader("🏆 Results")

# Row 1
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    image_result_1 = Image.open('./images/learning_curves_milestones.png')
    st.image(image_result_1, caption='Learning curves milestones', use_container_width=True)

with row1_col2:
    image_result_3 = Image.open('./images/per_class_f1_comparison.png')
    st.image(image_result_3, caption='Per class F1 comparison', use_container_width=True)

# Add spacing
st.write("")  # or st.markdown("<br>", unsafe_allow_html=True)

# Row 2
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    image_result_2 = Image.open('./images/confusion_matrices_with_shared_counts_percentage.png')
    st.image(image_result_2, caption='Confusion matrices in %', use_container_width=True)

with row2_col2:
    image_result_4 = Image.open('./images/image_model_table.png')
    st.image(image_result_4, caption='Model results comparison', use_container_width=True)

st.write("---")

# ============================================================================
# SAMPLE PREPROCESSED IMAGES GALLERY
# ============================================================================

st.subheader("📸 Sample Preprocessed Images")

st.write("""
Below are examples of preprocessed images from our dataset, organized by background type.
All images have been cropped and standardized to 500×500 pixels.
""")

preprocessed_dir = Path('./../data/preprocessed/Example_Preprocessed_Images')

if preprocessed_dir.exists():
    
    st.markdown("### ⬜ Image Before & After Samples")
    
    white_bg_images = list(preprocessed_dir.glob('*.png'))[:3]
    
    if len(white_bg_images) > 0:
        cols = st.columns(3)
        for idx, img_path in enumerate(white_bg_images):
            with cols[idx % 3]:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading {img_path.name}")
    else:
        st.info("No white background samples available yet.")

st.write("---")

# ============================================================================
# KEY FINDINGS
# ============================================================================

st.subheader("🔍 Key Findings")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **✅ What We Discovered:**
    - All images pre-standardized to 500×500px
    - 89.2% have white backgrounds (professional photography)
    - Only 0.01% completely blank images (very clean dataset)
    - Adaptive cropping preserves product content
    """)

with col2:
    st.info("""
    **📝 Preprocessing Impact:**
    - Background removal focuses on product
    - SSIM duplicate detection efficient and accurate
    - Category-based organization for training
    """)