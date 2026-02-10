# imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import random

# Simple page config
st.set_page_config(
    page_title="Data Exploration",
    page_icon="🔧",
    layout="wide"
)

# st.warning("⚠️ **Work in Progress** - This page is still under active development")

st.title("🔧 Data Exploration")

# ========== CHALLENGES ==========
st.subheader("⚠️ Key Challenges")

challenge_col1, challenge_col2 = st.columns(2)

with challenge_col1:
    st.markdown("**Data Quality Issues:**")
    st.write("""
    - 🌍 **Multilingual text**: 30+ languages (60% French, 20% German, 20% others)
    - 📉 **Missing data**: 35% products have no description
    - 📏 **High text variance**: variance in descriptions from 0 till 20000 words
    - 🏷️ **HTML artifacts**: Descriptions contain HTML tags and special characters
    """)

with challenge_col2:
    st.markdown("**Dataset Imbalance:**")
    st.write("""
    - ⚖️ **Class imbalance**: 13.4:1 ratio (largest vs smallest class)
    - 🔄 **Duplicates**: 1,414 duplicate products found
    - 🖼️ **Image variety**: Different backgrounds, angles, lighting
    - 🎨 **White margins**: Many images have excessive whitespace
    """)

st.write("---")

# ========== DATASET ==========
st.subheader("📦 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: #1f77b4; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Training Samples</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">84,916</h2>
        <p style="margin: 0; font-size: 16px; color: white;">Products to classify</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #ff7f0e; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Product Categories</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">27</h2>
        <p style="margin: 0; font-size: 16px; color: white;">Classification targets</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #2ca02c; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Image Data</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">2.2 GB</h2>
        <p style="margin: 0; font-size: 16px; color: white;">500×500 pixel images</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #9467bd; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <p style="margin: 0; font-size: 14px; color: white;">Text Data</p>
        <h2 style="margin: 10px 0; font-size: 42px; color: white;">60 MB</h2>
        <p style="margin: 0; font-size: 16px; color: white;">description + designation</p>
    </div>
    """, unsafe_allow_html=True)

st.info("""
**Features Available:**
- **Designation**: Product title
- **Description**: Product details
- **Image**: Product photo (500×500 pixels)
- **Product type code**: Product category (Target variable)
""")


st.write("---")

# Set matplotlib style for better visibility
plt.style.use('default')
sns.set_palette("husl")

# Load and merge data
@st.cache_data
def load_data():
    try:
        X_train = pd.read_csv('./../data/raw/X_train.csv')
        Y_train = pd.read_csv('./../data/raw/y_train.csv')
        df = pd.merge(X_train, Y_train, left_index=True, right_index=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.error("Cannot load data files. Please check that X_train.csv and Y_train.csv exist in ./data/text/")
    st.stop()

################ Dataset Overview Backup ################
# st.subheader("📊 Dataset Overview")

# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total Products", f"{len(df):,}")
# col2.metric("Categories", df['prdtypecode'].nunique())
# col3.metric("Features", len(df.columns) - 1)
# col4.metric("Image Size", "500x500")

#########################################################

# Show sample data #
st.write("**Sample of Actual Data (after merge of train & test set):**")
st.dataframe(df[['designation', 'description', 'prdtypecode']].head(5), use_container_width=True)

st.write("---")

# 1. Distribution of Target Variable
st.subheader("1️⃣ Distribution of Target Variable")

target_counts = df['prdtypecode'].value_counts().sort_index()

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(14, 6))

# Create bar plot
bars = ax.bar(range(len(target_counts)), target_counts.values, 
               color=plt.cm.viridis(target_counts.values / target_counts.values.max()))

# Set labels
ax.set_xlabel('Product Type Code', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of the prdtypecode - 27 product type codes', fontsize=14)

# Set x-axis ticks
ax.set_xticks(range(len(target_counts)))
ax.set_xticklabels(target_counts.index, rotation=0)

# Add average line
avg_count = len(df) / df['prdtypecode'].nunique()
ax.axhline(y=avg_count, color='red', linestyle='--', linewidth=2, 
           label=f'Average ({avg_count:.0f})')

# Add grid
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
st.pyplot(fig)

# Metrics
col1, col2, col3 = st.columns(3)
most_common = target_counts.idxmax()
least_common = target_counts.idxmin()
#col1.metric("Most Common", f"{most_common}", 
#            f"{(target_counts[most_common]/len(df)*100):.1f}%")
#col2.metric("Least Common", f"{least_common}", 
#            f"{(target_counts[least_common]/len(df)*100):.1f}%")
#imbalance_ratio = target_counts[most_common] / target_counts[least_common]
#col3.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}:1")

imbalance_ratio = target_counts[most_common] / target_counts[least_common]
col1.metric("Most Common", f"{most_common}", 
            f"{target_counts[most_common]:,} samples")  # Absolute number with comma separator
col2.metric("Least Common", f"{least_common}", 
            f"{target_counts[least_common]:,} samples")  # Absolute number with comma separator
col3.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}:1",
            f"Largest / Smallest")  # Added delta explanation

st.write("---")

# 2. Missing Values per Category
st.subheader("2️⃣ Missing Descriptions per Category")

st.info("""
💡 **Problem Identified**: 35% of products had missing descriptions, which would severely limit text-based classification.

**Our Solution**: By merging `designation` (product title) with `description` we created complete text data. 
Since designations were always present and contained key product information, this recovered the missing data.
""")

missing_by_category = df.groupby('prdtypecode').apply(
    lambda x: pd.Series({
        'Missing_Designation': (x['designation'].isna().sum() / len(x) * 100),
        'Missing_Description': (x['description'].isna().sum() / len(x) * 100),
        'Count': len(x)
    })
).reset_index()

# Sort by missing description percentage
missing_by_category = missing_by_category.sort_values('prdtypecode')

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(14, 6))

x = range(len(missing_by_category))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], missing_by_category['Missing_Designation'], 
               width, label='Missing Designation', color='lightblue')
bars2 = ax.bar([i + width/2 for i in x], missing_by_category['Missing_Description'], 
               width, label='Missing Description', color='coral')

ax.set_xlabel('Product Type Code', fontsize=12)
ax.set_ylabel('Missing Percentage (%)', fontsize=12)
ax.set_title('Missing Values by Product Category (Before Merging)', fontsize=14)  # ← Added "Before Merging"
ax.set_xticks(x)
ax.set_xticklabels(missing_by_category['prdtypecode'], rotation=0)

# Add 50% threshold line
ax.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50% threshold')

ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

overall_missing_desc = (df['description'].isna().sum() / len(df) * 100)
overall_missing_desig = (df['designation'].isna().sum() / len(df) * 100)

col1, col2, col3 = st.columns(3)
col1.metric("Missing Descriptions (Before)", f"{overall_missing_desc:.1f}%",
            f"{df['description'].isna().sum():,} products")
col2.metric("Missing Designations (Before)", f"{overall_missing_desig:.1f}%",
            f"{df['designation'].isna().sum():,} products")
col3.metric("Missing Text (After Merge)", "0.0%", 
            delta="✅ Solved!", delta_color="normal")

categories_over_50 = (missing_by_category['Missing_Description'] > 50).sum()
if categories_over_50 > 0:
    st.warning(f"⚠️ **Categories over 50% missing**: {categories_over_50} categories had >50% missing descriptions before merging!")

st.success("""
✅ **Result**: By merging designation + description, we achieved **0% missing text** data.
This gave us complete textual information for all products!
""")

st.write("---")

# 3. Textual Metrics
st.subheader("3️⃣ Textual Metrics Statistics")

st.info("""
💡 **Analysis Goal**: Understanding the characteristics of designation vs description fields 
to inform our text preprocessing strategy.
""")

# Calculate text metrics
df['designation_length'] = df['designation'].fillna('').astype(str).apply(len)
df['description_length'] = df['description'].fillna('').astype(str).apply(len)
df['designation_words'] = df['designation'].fillna('').astype(str).apply(lambda x: len(x.split()))
df['description_words'] = df['description'].fillna('').astype(str).apply(lambda x: len(x.split()))

# Create figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Characters plot
metrics_chars = ['designation_length', 'description_length']
means_chars = [df[m].mean() for m in metrics_chars]
labels_chars = ['Designation', 'Description']

axes[0].bar(labels_chars, means_chars, color=['lightblue', 'coral'])
axes[0].set_ylabel('Average Characters', fontsize=11)
axes[0].set_title('Average Character Length (Separate Fields)', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(means_chars):
    axes[0].text(i, v + max(means_chars)*0.02, f'{v:.0f}', 
                ha='center', va='bottom', fontweight='bold')

# Words plot
metrics_words = ['designation_words', 'description_words']
means_words = [df[m].mean() for m in metrics_words]
labels_words = ['Designation', 'Description']

axes[1].bar(labels_words, means_words, color=['lightblue', 'coral'])
axes[1].set_ylabel('Average Words', fontsize=11)
axes[1].set_title('Average Word Count (Separate Fields)', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(means_words):
    axes[1].text(i, v + max(means_words)*0.02, f'{v:.0f}', 
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
st.pyplot(fig)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Designation Chars", f"{df['designation_length'].mean():.0f}")
col2.metric("Avg Description Chars", f"{df['description_length'].mean():.0f}")
col3.metric("Avg Designation Words", f"{df['designation_words'].mean():.0f}")
col4.metric("Avg Description Words", f"{df['description_words'].mean():.0f}")

st.warning("""
⚠️ **Key Findings:**
- Designations are **short** (70 chars, 12 words) but **always present** (0% missing)
- Descriptions are **longer** (525 chars, 80 words) but **often missing** (35% missing)
- Combined: ~92 words average
""")

st.success("""
✅ **Preprocessing Decision**: We merged designation + description into a single text field.
- Recovers 35% missing data (designation fills gaps)
- Creates richer text (~92 vs 12-80 )
""")

st.write("---")

# 4. Language Statistics
st.subheader("4️⃣ Language Distribution")

st.write("**Detected Languages in Original Dataset:**")

# Actual data from Step_1_data_analysis.ipynb (language_designation analysis)
lang_data = pd.DataFrame({
    'Language': [
        'French (Français)',
        'English', 
        'German (Deutsch)',
        'Dutch (Nederlands)',
        'Catalan (Català)',
        'Italian (Italiano)',
        'Romanian (Română)',
        'Portuguese (Português)',
        'Spanish (Español)',
        'Indonesian (Bahasa Indonesia)',
        'Others'  # Sum of remaining languages
    ],
    'Count': [
        51707,  # French
        19625,  # English
        5020,   # German
        2170,   # Dutch
        1616,   # Catalan
        1136,   # Italian
        719,    # Romanian
        532,    # Portuguese
        517,    # Spanish
        286,    # Indonesian
        1588    # Others (sum of remaining ~20 languages)
    ]
})

# Show top 7 + Others for cleaner visualization
lang_data_display = lang_data.head(8).copy()  # Top 7 + Others combined

# Create matplotlib figure
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.Blues(range(50, 250, int(200/len(lang_data_display))))
bars = ax.bar(lang_data_display['Language'], lang_data_display['Count'], color=colors)

ax.set_xlabel('Language', fontsize=12)
ax.set_ylabel('Number of Products', fontsize=12)
ax.set_title('Language Distribution Across Products (Detected from Designation)', fontsize=14)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# Calculate metrics
total_products = lang_data['Count'].sum()  # 84,916
french_count = 51707
french_pct = (french_count / total_products * 100)
total_languages = 31  # From your notebook (30+ languages detected)

col1, col2, col3 = st.columns(3)
col1.metric("Dominant Language", "French 🇫🇷", 
            f"{french_count:,} products ({french_pct:.1f}%)")
col2.metric("Total Languages", f"{total_languages}")
col3.metric("Translation Target", "English 🇬🇧")

st.warning(f"**Challenge**: {total_languages} different languages require translation to English for unified processing!")

st.info("""
💡 **Data Source**: Language detection performed using spaCy with langdetect on the designation field.
Analysis from `Step_1_data_analysis.ipynb`.
""")

st.write("---")

# 5. Product Images Display
st.subheader("5️⃣ Random Product Images by Category")

image_folder = '../data/raw/images/'

if not os.path.exists(image_folder):
    st.error(f"Image folder not found: {image_folder}")
else:
    # Get unique categories
    categories = sorted(df['prdtypecode'].unique())
    
    # Initialize session state for random seed
    if 'image_seed' not in st.session_state:
        st.session_state.image_seed = 0
    
    # Create refresh button
    if st.button("🔄 Show Different Random Images", type="primary"):
        st.session_state.image_seed += 1
    
    # Use seed for reproducible random selection
    random.seed(st.session_state.image_seed)
    
    # Select 10 random categories to display
    display_categories = random.sample(list(categories), min(10, len(categories)))
    
    st.write(f"**Random Product Images:**")
    
    # Create 5 columns for 2 rows
    cols = st.columns(5)
    
    for idx, category in enumerate(display_categories):
        col_idx = idx % 5
        
        # Get products from this category
        category_products = df[df['prdtypecode'] == category]
        
        if len(category_products) > 0:
            # Select random product
            random_product = category_products.sample(1).iloc[0]
            
            # Construct image path - try different possible formats
            image_id = int(random_product['imageid'])
            product_id = int(random_product['productid'])
            
            # Try different possible filename formats
            possible_filenames = [
                f"image_{image_id}_product_{product_id}.jpg",
                f"{image_id}.jpg",
                f"image_{image_id}.jpg"
            ]
            
            image_path = None
            for filename in possible_filenames:
                test_path = os.path.join(image_folder, filename)
                if os.path.exists(test_path):
                    image_path = test_path
                    break
            
            with cols[col_idx]:
                st.write(f"**{category}**")
                
                if image_path and os.path.exists(image_path):
                    try:
                        img = Image.open(image_path)
                        st.image(img, use_container_width=True)
                    except:
                        st.write("❌ Error loading")
                else:
                    st.write("📷 Image not found")
    st.write("")
    st.warning("""
    ⚠️ **Image Challenges Identified:**
    - **White margins**: Many product images contain excessive whitespace that needed cropping
    - **Variable backgrounds**: Inconsistent backgrounds (white, colored, textured) across products
    - **Pixel distribution**: Mix of photos with varying distribution
    - **Blank images**: Some images are completely blank or corrupted (requires removal)
    """)

# Debug info for images path
st.write("---")

# Key Takeaways
st.subheader("🎯 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **✅ Dataset Strengths:**
    - Large real-world dataset provides sufficient data for multi-modal training (84,916 products)
    - Clear target category structure (27 classes)
    - Consistent image dimensions (500×500)
    """)

with col2:
    st.error("""
    **❌ Major Challenges:**
    
    **Text Issues:**
    - **Data cleaning & merging**: Missing descriptions, HTML tags, duplicates
    - **Language standardization**: Multiple languages requiring translation to English
        
    **Image Issues:**
    - wide range in quality (background, whitespaces, pixel distribution)
    - some blank/corrupted images needing removal
    """)

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **💡 Strategic Insights**
    - **Text is a huge bottleneck**: Short descriptions + missing data + multilingual content
    - → *Solution*: Merge designation + description, translate to English
    
    - **Class imbalance is significant but manageable**: 13.4:1 ratio 
    - → *Solution*: Class weights during training
    """)

with col2:
    st.info("""
    **💡 Strategic Insights**
    - **Multimodal fusion potential**: Combining text and image features can achieve 80%+ accuracy 
    - → *Solution*: Transfer learning (BERT + EfficientNet) with late fusion

    - **Data quality over quantity**: Text cleaning lead to 0% missing values 
    - → *Result*: High-quality samples for transfer learning
    """)

st.write("---")
st.caption("All statistics calculated from actual Rakuten training dataset (84,916 samples)")