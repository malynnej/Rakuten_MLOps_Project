import streamlit as st

# Simple page config
st.set_page_config(page_title="Transfer Learning & Fusion", layout="wide")

# st.warning("⚠️ **Work in Progress** - This page is still under active development")

st.title("Transfer Learning & Multimodal Fusion")


# --- SECTION 1: BERT Text Architecture and Results ---
st.header("BERT Text Architecture and Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Architechture")
    st.image("./images/bert_architechture.png")
    st.caption("simplified architechture overview")

with col2:
    st.subheader("BERT Results Table")
    st.image("./images/bert_results.png")
    st.caption("satisfactory results after 3 runs ~85%")

st.divider()



# --- SECTION 2: BERT Text Encoder ---
st.header("BERT Text Encoder Training")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Encoder Phase")
    st.image("./images/training_history_bert_encoder.png")
    st.caption("7 epochs, validation accuracy ~85%")

with col2:
    st.subheader("Optimized Phase")
    st.image("./images/training_history_bert_optim.png")
    st.caption("18 epochs with fine-tuning, validation accuracy ~85%")

st.markdown("BERT processes text descriptions (max 128 tokens) and extracts 768-dimensional semantic features. Trained standalone to reach **85% test accuracy** with **84% Macro F1-Score**.")

st.divider()

# --- SECTION 3: BERT Confusion Matrices ---
st.header("BERT Classification Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Encoder Confusion Matrix")
    st.image("./images/fusion_bert_encoder.png")

with col2:
    st.subheader("Optimized Confusion Matrix")
    st.image("./images/fusion_bert_optim.png")

st.divider()

# --- SECTION 4: Architecture Overview ---
st.header("Late Fusion Architecture")

col1, col2 = st.columns([1, 1])

with col1:
    st.image("./images/multimodal_image.png", caption="Late Fusion Model Architecture")

with col2:
    st.markdown("""
    **Feature Extraction:**
    - **EfficientNet-B0:** 1280-dim visual features (global avg pooling)
    - **BERT:** 768-dim text features ([CLS] pooler, max 128 tokens)
    - **Concatenated:** 2048-dim multimodal representation
    
    **Fusion Classifier Head (progressive reduction):**
    - 2048 → 1024: BatchNorm + ReLU + Dropout(0.3)
    - 1024 → 512: BatchNorm + ReLU + Dropout(0.2)
    - 512 → 128: ReLU
    - 128 → 27: Output layer
    """)

st.divider()

# --- SECTION 5: Multimodal Fusion Training ---
st.header("Multimodal Fusion Training")

st.image("./images/training_history_fusion_model.png", caption="Two-Phase Training Strategy")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Phase 1 (Epochs 1-5): Frozen Encoders**
    - Only fusion head trained
    - AdamW optimizer, lr=1e-4
    - ReduceLROnPlateau scheduler
    - Reached ~86% validation accuracy
    """)

with col2:
    st.markdown("""
    **Phase 2 (Epochs 6+): Selective Fine-Tuning**
    - Top 50 EfficientNet layers unfrozen
    - Last 2 BERT encoder layers + pooler unfrozen
    - Discriminative learning rates (head: 1e-4, encoders: 1e-5)
    - Early stopping at epoch 7 → **87.4% val accuracy**
    """)

st.divider()

# --- SECTION 6: Fusion Results ---
st.header("Fusion Model Results")

col1, col2 = st.columns([2, 1])

with col1:
    st.image("./images/fusion_cm.png", caption="Late Fusion Confusion Matrix")

with col2:
    st.markdown("""
    **Test Performance:**
    - Accuracy: **85%**
    - Macro F1-Score: **83%**
    
    **Per-Class F1 Range:**
    - Lowest: 54% (minority class 1280)
    - Highest: 97% (classes 1160, 2583)
    
    **Data Augmentation Applied:**
    - Random flip, rotation (30°)
    - ColorJitter, Gaussian blur
    - Random erasing (p=0.25)
    - Weighted random sampler
    """)

st.divider()

# --- SECTION 7: Model Comparison ---
st.header("Model Comparison")

st.markdown("""
| Model | Input | Test Accuracy | Macro F1-Score |
|-------|-------|---------------|----------------|
| BERT | Text | 85% | 84% |
| EfficientNet-B0 | Image | 63% | 59% |
| **Late Fusion** | **Text + Image** | **85%** | **83%** |
""")

st.info("The late fusion strategy combines complementary learnings from both modalities, achieving the best overall performance for Rakuten product classification.")