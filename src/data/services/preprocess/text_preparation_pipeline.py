import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import pickle
import json
import time
from datetime import timedelta

from services.preprocess.text_cleaning import TextCleaning
from services.preprocess.text_outliers import TransformTextOutliers
from core.config import load_config, get_path

class TextPreparationPipeline:
    """
    Unified preprocessing pipeline with:
    - Text cleaning (HTML removal, encoding fixes)
    - Outlier transformation (redundant text removal, summarization)
    - Label encoding (categorical → integers for BERT)
    - Data splitting (train/val/test/holdout)
    - Tokenization (BERT tokenizer with padding/truncation)
    - Saving as parquet files
    
    Handles:
    - Batch processing (files for training)
    """
    def __init__(self):
        """
        Initialize pipeline components.
        Loads configs and sets up tokenizer, text cleaner, outlier transformer.
        """
        # Load configs
        self.paths_config = load_config("paths")
        self.preproc_config = load_config("params")
        
        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.preproc_config["preprocessing"]["bert_model"]
        )
        
        # Label encoder (will be fit during batch processing)
        self.label_encoder = None
        
        # Initialize text cleaning component
        self.text_cleaner = TextCleaning(
            html_to_text=self.preproc_config["preprocessing"].get("html_to_text", True),
            words_encoding=self.preproc_config["preprocessing"].get("words_encoding", True)
        )
        
        # Initialize outlier transformer
        self.outlier_transformer = TransformTextOutliers(
            column_to_transform="text",
            model=self.preproc_config["preprocessing"].get("llm_model", "en_core_web_sm"),
            word_count_threshold=self.preproc_config["preprocessing"].get("word_count_threshold", 250),
            sentence_normalization=self.preproc_config["preprocessing"].get("sentence_normalization", True),
            similarity_threshold=self.preproc_config["preprocessing"].get("similarity_threshold", 0.8),
            factor=self.preproc_config["preprocessing"].get("factor", 3)
        )
        
        print("PreprocessingPipeline initialized successfully!")

    def preprocess_batch(self, df: pd.DataFrame, retrain: bool = False, save_holdout: bool = True) -> dict:
        """
        Process entire dataframe for training.
        
        Steps:
        1. Combine designation + description columns
        2. Split data EARLY (holdout gets raw data)
        3. Clean text (HTML, encoding) - ONLY on train/val/test
        4. Transform outliers - ONLY on train/val/test
        5. Encode labels (categories → integers)
        6. Tokenize train/val/test
        7. Save as parquet files
        
        Holdout set receives:
        - Combined text (designation + description)
        - Original raw text (no cleaning/transformation)
        - Labels (for evaluation)
        - NOT tokenized (done at prediction time)
        
        Input:
            df (pd.DataFrame): Raw data with columns:
                - designation: Product title
                - description: Product description
                - prdtypecode: Category ID
            retrain (bool): If True, loads existing label encoder
            save_holdout (bool): If True, saves holdout set for future retraining
        
        Returns:
            dict: {
                "train": path to train.parquet,
                "val": path to val.parquet,
                "test": path to test.parquet,
                "holdout": path to holdout.parquet (if save_holdout=True),
                "num_train": int,
                "num_val": int,
                "num_test": int,
                "num_holdout": int (if save_holdout=True),
                "label_encoder_path": str,
                "num_labels": int
            }
        """
        t_start = time.time()
        
        print("=" * 60)
        print("BATCH PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Retrain mode: {retrain}")
        print(f"Save holdout: {save_holdout}")
        print(f"Input data shape: {df.shape}")
        
        # 1. Combine text columns (designation + description)
        print("\n>>> Step 1: Combining text columns")
        df = self._combine_text_columns(df)
        
        # 2. EARLY SPLIT: Separate holdout BEFORE preprocessing
        print("\n>>> Step 2: Splitting data (holdout gets raw data)")
        if save_holdout:
            main_df, holdout_df = self._split_holdout_early(df)
        else:
            main_df = df
            holdout_df = None
        
        # 3. Text cleaning (HTML → plain text, fix encoding) - ONLY main data
        print("\n>>> Step 3: Text cleaning (train/val/test only)")
        main_df, _ = self.text_cleaner.cleanTxt(main_df, ["text"])
        
        # Handle empty texts
        main_df["text"] = main_df["text"].apply(lambda x: x if (x and x.strip() != "") else "[EMPTY]")
        
        # 4. Text outlier transformation - ONLY main data
        if self.preproc_config["preprocessing"].get("transform_outliers", True):
            print("\n>>> Step 4: Text outlier transformation (train/val/test only)")
            main_df, _, _ = self.outlier_transformer.transform_outliers(main_df)
        else:
            print("\n>>> Step 4: Skipping outlier transformation (disabled in config)")
        
        # 5. Label encoding (prdtypecode → integer labels)
        print("\n>>> Step 5: Label encoding")
        main_df, label_encoder_path, num_labels = self._encode_labels(main_df, retrain)
        
        # Also encode holdout labels (but don't preprocess the text)
        if holdout_df is not None:
            holdout_df["labels"] = self.label_encoder.transform(holdout_df["prdtypecode"].values)
        
        # 6. Split main data into train/val/test
        print("\n>>> Step 6: Splitting train/val/test")
        train_df, val_df, test_df = self._split_data(main_df)
        
        # 7. Tokenization (BERT tokenizer) - ONLY train/val/test
        print("\n>>> Step 7: Tokenizing train/val/test (holdout stays raw)")
        train_df = self._tokenize_dataframe(train_df)
        val_df = self._tokenize_dataframe(val_df)
        test_df = self._tokenize_dataframe(test_df)
        
        # 8. Save as parquet files
        print("\n>>> Step 8: Saving preprocessed data")
        output_paths = self._save_splits(train_df, val_df, test_df, holdout_df)
        
        # Add metadata to output
        output_paths["label_encoder_path"] = str(label_encoder_path)
        output_paths["num_labels"] = num_labels
        
        t_end = time.time()
        t_exec = str(timedelta(seconds=t_end - t_start))
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Execution time: {t_exec}")
        print(f"Label encoder: {label_encoder_path}")
        print(f"Number of classes: {num_labels}")
        print(f"Output directory: {get_path('data.preprocessed')}")
        
        if holdout_df is not None:
            print("\n📦 Holdout set saved with RAW text (not preprocessed)")
            print("   → Use for realistic prediction/retraining simulation")
        
        return output_paths

    
    def _encode_labels(self, df: pd.DataFrame, retrain: bool) -> tuple:
        """
        Encode categorical labels (prdtypecode) to integers.
        
        For initial training:
        - Creates new LabelEncoder
        - Fits on all categories
        - Saves encoder and mappings to disk
        
        For retraining:
        - Loads existing LabelEncoder
        - Transforms using existing mappings
        - Raises error if new categories appear
        
        Returns:
            df (pd.DataFrame): DataFrame with 'labels' column added
            encoder_path (Path): Path to saved label encoder
            num_labels (int): Number of unique classes
        """
        models_dir = get_path("models.save_dir")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        encoder_path = models_dir / "label_encoder.pkl"
        mapping_path = models_dir / "label_mappings.json"
        
        if retrain:
            # RETRAINING: Load existing encoder
            if not encoder_path.exists():
                raise FileNotFoundError(
                    f"Label encoder not found at {encoder_path}. "
                    f"Cannot retrain without existing encoder!"
                )
            
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            
            print(f"Loaded existing label encoder")
            print(f"  Classes: {len(self.label_encoder.classes_)}")
            print(f"  Categories: {sorted(self.label_encoder.classes_)[:10]}...")
            
            # Transform with existing encoder
            try:
                df["labels"] = self.label_encoder.transform(df["prdtypecode"].values)
            except ValueError as e:
                unique_categories = df["prdtypecode"].unique()
                unknown_categories = set(unique_categories) - set(self.label_encoder.classes_)
                raise ValueError(
                    f"New categories detected in data!\n"
                    f"Unknown categories: {unknown_categories}\n"
                    f"Existing encoder only supports: {set(self.label_encoder.classes_)}"
                )
        
        else:
            # INITIAL TRAINING: Create new encoder
            self.label_encoder = LabelEncoder()
            df["labels"] = self.label_encoder.fit_transform(df["prdtypecode"].values)
            
            print(f"Created new label encoder")
            print(f"  Classes: {len(self.label_encoder.classes_)}")
            print(f"  Categories: {sorted(self.label_encoder.classes_)}")
            
            # Save label encoder
            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)
            print(f"\nSaved label encoder to: {encoder_path}")
            
            # Create and save label mappings (for BERT model config)
            id2label = {i: str(cat_id) for i, cat_id in enumerate(self.label_encoder.classes_)}
            label2id = {str(cat_id): i for i, cat_id in enumerate(self.label_encoder.classes_)}
            
            mappings = {
                "id2label": id2label,
                "label2id": label2id,
                "num_labels": len(self.label_encoder.classes_)
            }
            
            with open(mapping_path, "w") as f:
                json.dump(mappings, f, indent=2)
            print(f"Saved label mappings to: {mapping_path}")
            
            # Print sample mappings
            print("\n" + "-" * 40)
            print("LABEL MAPPINGS (sample)")
            print("-" * 40)
            sample_items = list(id2label.items())[:5]
            for idx, cat in sample_items:
                print(f"  {idx} → Category {cat}")
            print(f"  ... ({len(id2label) - 5} more)")
        
        num_labels = len(self.label_encoder.classes_)
        
        return df, encoder_path, num_labels
    
    def _combine_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine 'designation' and 'description' into single 'text' column.
        
        Logic:
        - If description == designation: use only designation (avoid duplication)
        - Otherwise: concatenate with "; " separator
        
        Returns:
            df with 'text' column added
        """
        df = df.copy()
        
        # Fill NaN with empty string
        df["description"] = df["description"].fillna("")
        df["designation"] = df["designation"].fillna("")
        
        # Create mask for rows where description equals designation
        mask_equal = df["description"] == df["designation"]
        
        # Vectorized combination
        df["text"] = df["designation"] + "; " + df["description"]
        
        # For equal rows, use only designation
        df.loc[mask_equal, "text"] = df.loc[mask_equal, "designation"]
        
        print(f"  Combined designation + description → text column")
        
        return df
    
    def _split_holdout_early(self, df: pd.DataFrame) -> tuple:
        """
        Split holdout set BEFORE any preprocessing.
        
        This ensures holdout data remains "raw" and realistic for:
        - Testing prediction pipeline (text must go through full preprocessing)
        - Simulating retraining (new data arrives unprocessed)
        
        Returns:
            (main_df, holdout_df)
            - main_df: will be preprocessed and split into train/val/test
            - holdout_df: raw text, only combined designation+description
        """
        holdout_size = self.preproc_config["preprocessing"].get("holdout_size", 0.10)
        random_state = self.preproc_config["preprocessing"].get("random_state", 42)
        
        # Temporary label encoding just for stratification
        temp_encoder = LabelEncoder()
        temp_labels = temp_encoder.fit_transform(df["prdtypecode"].values)
        
        main_df, holdout_df = train_test_split(
            df,
            test_size=holdout_size,
            random_state=random_state,
            stratify=temp_labels
        )
        
        print(f"  Main data: {len(main_df)} ({len(main_df)/len(df)*100:.1f}%) [will be preprocessed]")
        print(f"  Holdout:   {len(holdout_df)} ({len(holdout_df)/len(df)*100:.1f}%) [stays RAW]")
        
        return main_df, holdout_df

    def _split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train/val/test sets.
        
        Split strategy:
        - Test: 15% (for final evaluation)
        - Val: 15% of remaining (for validation during training)
        - Train: remainder (~70%)
        
        Uses stratification on labels to maintain class balance.
        Uses random_state=42 for reproducibility.
        
        Returns:
            (train_df, val_df, test_df)
        """
        test_size = self.preproc_config["preprocessing"].get("test_size", 0.15)
        val_size = self.preproc_config["preprocessing"].get("val_size", 0.15)
        random_state = self.preproc_config["preprocessing"].get("random_state", 42)
        
        # First split: (train+val) vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["labels"]
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df["labels"]
        )
        
        print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def _tokenize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenize all texts in dataframe using BERT tokenizer.
        
        Tokenization settings (same as training!):
        - max_length: from config (default 128)
        - padding: "max_length" (all sequences same length)
        - truncation: True (cut long texts)
        
        Adds columns:
        - input_ids: token IDs for BERT
        - attention_mask: padding mask for BERT
        
        Returns:
            df with tokenized columns added
        """
        df = df.copy()
        
        # Batch tokenize for efficiency
        tokenized = self.tokenizer(
            df["text"].tolist(),
            max_length=self.preproc_config["preprocessing"]["max_length"],
            padding="max_length",
            truncation=True
        )
        
        # Add tokenized columns
        df["input_ids"] = tokenized["input_ids"]
        df["attention_mask"] = tokenized["attention_mask"]
        
        print(f"  Tokenized {len(df)} samples")
        print(f"  Max length: {self.preproc_config['preprocessing']['max_length']}")
        
        return df
    
    def _save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                 test_df: pd.DataFrame, holdout_df: pd.DataFrame = None) -> dict:
        """
        Save train/val/test/holdout splits as parquet files.
        
        Train/Val/Test: Fully preprocessed and tokenized
        Holdout: Raw text with labels only (for realistic prediction testing)
        """
        preprocessed_dir = get_path("data.preprocessed")
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Define paths
        train_path = preprocessed_dir / "train.parquet"
        val_path = preprocessed_dir / "val.parquet"
        test_path = preprocessed_dir / "test.parquet"
        
        # Columns for preprocessed data
        columns_to_save = ["text", "input_ids", "attention_mask", "labels", "prdtypecode"]
        
        # Save main splits (fully preprocessed)
        train_df[columns_to_save].to_parquet(train_path, index=False)
        val_df[columns_to_save].to_parquet(val_path, index=False)
        test_df[columns_to_save].to_parquet(test_path, index=False)
        
        print(f"  Saved train → {train_path} [preprocessed + tokenized]")
        print(f"  Saved val   → {val_path} [preprocessed + tokenized]")
        print(f"  Saved test  → {test_path} [preprocessed + tokenized]")
        
        result = {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "num_train": len(train_df),
            "num_val": len(val_df),
            "num_test": len(test_df)
        }
        
        # Save holdout (RAW text, only labels added)
        if holdout_df is not None:
            holdout_path = preprocessed_dir / "holdout_raw.parquet"
            # Save only: text (raw), designation, description, labels, prdtypecode
            holdout_columns = ["designation", "description", "text", "labels", "prdtypecode"]
            holdout_df[holdout_columns].to_parquet(holdout_path, index=False)
            print(f"  Saved holdout → {holdout_path} [RAW TEXT - not preprocessed]")
            
            result["holdout"] = str(holdout_path)
            result["num_holdout"] = len(holdout_df)
        
        return result

    
# Standalone execution for testing
if __name__ == "__main__":
    import logging
    
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    # Load raw data
    raw_dir = get_path("data.raw")
    paths = load_config("paths")
    X_train_raw = paths["data"]["X_train_raw"]
    y_train_raw = paths["data"]["y_train_raw"]
    X_train = pd.read_csv(raw_dir / X_train_raw, index_col=0)
    y_train = pd.read_csv(raw_dir / y_train_raw, index_col=0)
    
    # Combine
    df = X_train.join(y_train)
    
    print(f"Loaded {len(df)} samples")
    
    # Preprocess
    pipeline = TextPreparationPipeline()
    output_paths = pipeline.preprocess_batch(df, retrain=False, save_holdout=True)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    for key, value in output_paths.items():
        print(f"{key}: {value}")