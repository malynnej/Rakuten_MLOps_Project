import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertTokenizer, AutoModelForSequenceClassification
from PIL import Image
import pandas as pd
import json
import random

#Simple page config
st.set_page_config(page_title="Live Prediction", layout="wide", page_icon="🎯")

# st.warning("⚠️ **Work in Progress** - This page is still under active development")

st.title("🎯 Live Prediction Demo")

# --- Model Definitions ---

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, effnet_path, dropout=0.5):
        super(EfficientNetClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(weights=None)
        self.num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        state_dict = torch.load(effnet_path, map_location="cpu")
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("base_model.", "")
            new_state[new_k] = v
        self.base_model.load_state_dict(new_state, strict=True)
        
    def forward(self, x):
        return self.base_model(x)
    
    def extract_features(self, x):
        x = self.base_model.features(x)
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def freeze_base(self):
        for param in self.base_model.parameters():
            param.requires_grad = False


class LateFusionModel(nn.Module):
    def __init__(self, bert_path, effnet_path, num_classes=27):
        super(LateFusionModel, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(bert_path, num_labels=27)
        self.bert_hidden = self.bert.config.hidden_size
        for p in self.bert.parameters():
            p.requires_grad = False

        self.effnet = EfficientNetClassifier(num_classes=27, effnet_path=effnet_path, dropout=0.5)
        self.effnet.freeze_base()
        self.effnet_hidden = self.effnet.num_features

        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden + self.effnet_hidden, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask, image):
        bert_output = self.bert.base_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = bert_output.pooler_output
        img_feat = self.effnet.extract_features(image)
        fused = torch.cat([text_feat, img_feat], dim=1)
        out = self.classifier(fused)
        return out

# --- Functions ---

@st.cache_resource
def load_model():
    """Load fusion model (cached)"""
    bert_path = "./../data_model/bert-model"
    effnet_path = "./../data_model/effnet-model/best_model_final.pth"
    fusion_path = "./../data_model/fusion-model/fusion_best_V2.pth"
    
    model = LateFusionModel(bert_path=bert_path, effnet_path=effnet_path, num_classes=27)
    model.load_state_dict(torch.load(fusion_path, map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    """Load BERT tokenizer (cached)"""
    bert_path = "./../data_model/bert-model"
    return BertTokenizer.from_pretrained(bert_path)

@st.cache_data
def load_label_mappings():
    with open("./../inference/label2id", "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label

@st.cache_data
def load_fusion_data():
    return pd.read_csv("./../data/train_fusion.csv")

def get_random_samples(df, n=3):
    return df.sample(n=n).reset_index(drop=True)

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def preprocess_text(text, tokenizer):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return encoded["input_ids"], encoded["attention_mask"]

def predict(model, tokenizer, text, image):
    input_ids, attention_mask = preprocess_text(text, tokenizer)
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, image=img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    return pred_idx, confidence


# --- Streamlit Page ---

# Load model and data
with st.spinner("Loading model..."):
    model = load_model()
    tokenizer = load_tokenizer()
    label2id, id2label = load_label_mappings()
    fusion_df = load_fusion_data()

st.subheader("Loading Fusion Model...")
st.success("Model loaded! Ready to predict 👌")

st.divider()

# --- Section 1: Sample Display ---
st.header("Sample Products from Dataset")
st.info("📋 Choose one of these samples, type in description & upload image and compare true with predicted class.")

if "samples" not in st.session_state:
    st.session_state.samples = get_random_samples(fusion_df, n=3)

samples = st.session_state.samples

col1, col2, col3 = st.columns(3)

for idx, col in enumerate([col1, col2, col3]):
    with col:
        st.subheader(f"Sample {idx + 1}")
        
        row = samples.iloc[idx]
        true_class_id = row['label']
        true_class = id2label[true_class_id]
        text = row['text']
        image_path = row['image'].replace("./data/", "../data/")
        
        st.metric("True Class", true_class)
        
        try:
            img = Image.open(image_path).convert("RGB")
            st.image(img, use_container_width=True)
            filename = image_path.split("/")[-1]
            st.caption(filename)
        except:
            st.warning(f"Image not found: {image_path}")
        
        st.text_area(f"Text {idx + 1}", value=text[:500] + "..." if len(text) > 500 else text, height=150, disabled=True, key=f"sample_text_{idx}")

st.divider()

# --- Section 2: Prediction Input ---
st.header("Make a Prediction")

col_input, col_result = st.columns([1, 1])

with col_input:
    text_input = st.text_area("Enter Product Text", height=150, placeholder="Type or paste the product description here...")
    
    uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])
    
    true_class = None
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", width=224)
        
        filename = uploaded_image.name
        mask = fusion_df['image'].str.contains(filename, regex=False)
        matches = fusion_df[mask]
        if len(matches) > 0:
            true_class_id = matches.iloc[0]['label']
            true_class = id2label[true_class_id]
            st.info(f"📎 True Class detected: **{true_class}**")
        else:
            st.warning("⚠️ Image not found in dataset - true class unknown")

    predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

with col_result:
    st.subheader("Prediction Result")
    
    # Reserve fixed space with empty container
    result_container = st.container()
    
    with result_container:
        if predict_btn:
            if not text_input.strip():
                st.error("Please enter product text!")
            elif not uploaded_image:
                st.error("Please upload an image!")
            else:
                with st.spinner("Predicting..."):
                    pred_idx, confidence = predict(model, tokenizer, text_input, image)
                    pred_class = id2label[pred_idx]
                
                st.markdown("---")
                
                col_true, col_pred = st.columns(2)
                
                with col_true:
                    if true_class:
                        st.metric(label="True Class", value=true_class)
                    else:
                        st.metric(label="True Class", value="Unknown")
                
                with col_pred:
                    st.metric(label="Predicted Class", value=pred_class)
                
                if true_class and true_class == pred_class:
                    st.success(f"✅ Correct! (Confidence: {confidence:.1%})")
                elif true_class and true_class != pred_class:
                    st.error(f"❌ Incorrect (Confidence: {confidence:.1%})")
                else:
                    st.info(f"Confidence: {confidence:.1%}")
        else:
            st.info("Enter text, upload image, and click Predict")
            st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

st.divider()

# --- Section 3: Sample Lookup ---
st.header("🔍 Sample Lookup")
st.info("Enter an image filename to find its corresponding text and class.")

filename_input = st.text_input(
    "Image Filename", 
    placeholder="e.g. image_1260388396_product_3893245817.jpg"
)

if st.button("🔎 Lookup", type="secondary"):
    if filename_input.strip():
        try:
            mask = fusion_df['image'].str.contains(filename_input.strip(), regex=False)
            matches = fusion_df[mask]
            
            if len(matches) > 0:
                row = matches.iloc[0]
                true_class_id = row['label']
                true_class = id2label[true_class_id]
                
                st.success(f"✅ Found!")
                
                st.metric("True Class", true_class)
                
                st.subheader("Text")
                st.code(row['text'])
                
                st.subheader("Image Path")
                st.code(row['image'])
            else:
                st.error("❌ No matching row found")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.warning("Please enter a filename")