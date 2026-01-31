"""
Text Preparation Pipeline with Retraining Support

Initial Training: 1. Only use provided raw data (no old preprocessed data) -
initial training

Handles 3 retraining cases: 1. Periodic new data (combine old preprocessed data
+ new raw data) - retraining with fine-tuning only 2. New classes detection
(with periodic approach) - initial training needed (re-fit of label encoder) 3.
Parameter changes (trigger retraining)
"""

import json
import pickle
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from core.config import get_path, load_config
from services.preprocess.text_cleaning import TextCleaning
from services.preprocess.text_outliers import TransformTextOutliers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


class TextPreparationPipeline:
    """
    Unified preprocessing pipeline with retraining support.

    Features: - Text cleaning (HTML removal, encoding fixes) - Outlier
    transformation (redundant text removal, summarization) - Label encoding with
    new class detection - Data splitting (train/val/test/holdout) - Tokenization
    (BERT tokenizer) - Combining new and old preprocessed data
    """

    def __init__(self):
        """
        Initialize pipeline components. Loads configs and sets up tokenizer,
        text cleaner, outlier transformer.
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
            words_encoding=self.preproc_config["preprocessing"].get("words_encoding", True),
        )

        # Initialize outlier transformer
        self.outlier_transformer = TransformTextOutliers(
            column_to_transform="text",
            model=self.preproc_config["preprocessing"].get("llm_model", "en_core_web_sm"),
            word_count_threshold=self.preproc_config["preprocessing"].get(
                "word_count_threshold", 250
            ),
            sentence_normalization=self.preproc_config["preprocessing"].get(
                "sentence_normalization", True
            ),
            similarity_threshold=self.preproc_config["preprocessing"].get(
                "similarity_threshold", 0.8
            ),
            factor=self.preproc_config["preprocessing"].get("factor", 3),
        )

        print("PreprocessingPipeline initialized successfully!")

    # ============================================ MAIN ENTRY POINTS
    # ============================================

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        combine_existing_data: bool = False,
        save_holdout: bool = True,
    ) -> dict:
        """
        Main entry point for preparing training data.

        Args:
            df: Raw dataframe with designation, description, prdtypecode
            combine_existing_data: If True, combines new raw data with
            preprocessed data save_holdout: If True, saves holdout set

        Returns:
            dict with paths and metadata
        """
        t_start = time.time()

        print("=" * 60)
        print("TEXT PREPARATION PIPELINE")
        print("=" * 60)
        print(
            f"Mode: {
                'Combine with preprocessed data' 
                if combine_existing_data 
                else 'only new raw data'
            }"
        )
        print(f"Input data: {len(df):,} samples")
        print(f"Save holdout: {save_holdout}")

        # Detect new classes
        has_new_classes, new_classes_info = self._detect_new_classes(df)

        if has_new_classes:
            print("\n  NEW CLASSES DETECTED!")
            print(f"   New classes: {new_classes_info['new_classes']}")
            print("   -> Re-fit label encoder")

        # Standard preprocessing pipeline
        output_paths = self._preprocess_pipeline(
            df,
            combine_existing_data=combine_existing_data,
            save_holdout=save_holdout,
            has_new_classes=has_new_classes,
        )

        # Add metadata
        output_paths["has_new_classes"] = has_new_classes
        if has_new_classes:
            output_paths["new_classes_info"] = new_classes_info

        t_end = time.time()
        t_exec = str(timedelta(seconds=t_end - t_start))

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Execution time: {t_exec}")
        if has_new_classes:
            print("New classes detected - label encoder re-fitted")

        return output_paths

    # ============================================ NEW CLASSES DETECTION
    # ============================================

    def _detect_new_classes(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Detect if new product classes exist in the data.

        Returns:
            (has_new_classes, info_dict) - has_new_classes: True if new classes
            found - info_dict: Details about new/existing classes
        """
        print("\n>>> Checking for new classes")

        models_dir = get_path("models.save_dir")
        encoder_path = models_dir / "label_encoder.pkl"

        if not encoder_path.exists():
            print("  No existing label encoder found, create label encoder")
            return True, {}

        # Load existing encoder
        with open(encoder_path, "rb") as f:
            existing_encoder = pickle.load(f)

        existing_classes = set(existing_encoder.classes_)
        new_data_classes = set(df["prdtypecode"].unique())

        # Find new classes
        unseen_classes = new_data_classes - existing_classes

        info = {
            "existing_classes": sorted(existing_classes),
            "new_data_classes": sorted(new_data_classes),
            "new_classes": sorted(unseen_classes),
            "num_existing": len(existing_classes),
            "num_new": len(unseen_classes),
            "num_total": len(existing_classes | new_data_classes),
        }

        if unseen_classes:
            print("  NEW CLASSES DETECTED:")
            print(f"     New classes: {sorted(unseen_classes)}")
            print(f"     Existing:    {len(existing_classes)} classes")
            print(f"     Total now:   {info['num_total']} classes")
            print("  Label encoder must be re-fitted")
            print("  Model must be retrained from scratch")
            return True, info

        print(f"  No new classes - all {len(existing_classes)} classes already known")
        return False, info

    def _preprocess_pipeline(
        self,
        df_new: pd.DataFrame,
        combine_existing_data: bool,
        save_holdout: bool,
        has_new_classes: bool = False,
    ) -> dict:
        """
        Core preprocessing pipeline.

        Steps: 1. Combine text columns (if not already done) 2. Label encoding
        3. Early Holdout Split 4. Clean text (HTML, encoding) 5. Transform
        outliers 6. Split new data (train, test, val) and combine with old data
        if defined 7. Tokenize 8. Save parquet files
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Combining mode: {combine_existing_data}")
        print(f"New classes: {has_new_classes}")
        print(f"Save holdout: {save_holdout}")
        print(f"Input data shape: {df_new.shape}")

        # Step 1: Combine text columns if not already done
        if "text" not in df_new.columns:
            print("\n>>> Step 1: Combining text columns")
            df_new = self._combine_text_columns(df_new)
        else:
            print("\n>>> Step 1: Text column already exists, skipping combination")

        # Step 2: Label encoding
        print("\n>>>  Step2: Label encoding")
        df_new, label_encoder_path, num_labels = self._handle_label_encoding(
            df_new, combine_existing_data, has_new_classes
        )

        # Step 3: EARLY SPLIT: Separate holdout BEFORE preprocessing for new
        # data
        print("\n>>> Step 3: Splitting holdout (holdout gets raw data)")
        if save_holdout:
            main_df_new, holdout_df_new = self._split_holdout_early(df_new)
        else:
            main_df_new = df_new
            holdout_df_new = None

        # Step 4: Text cleaning (HTML -> plain text, fix encoding)
        print("\n>>> Step 4: Text cleaning (train/val/test only)")
        main_df_new, _ = self.text_cleaner.cleanTxt(main_df_new, ["text"])

        # Handle empty texts
        main_df_new["text"] = main_df_new["text"].apply(
            lambda x: x if (x and x.strip() != "") else "[EMPTY]"
        )

        # Step 5: Text outlier transformation
        if self.preproc_config["preprocessing"].get("transform_outliers", True):
            print("\n>>> Step 5: Text outlier transformation (train/val/test only)")
            main_df_new, _, _ = self.outlier_transformer.transform_outliers(main_df_new)
        else:
            print("\n>>> Step 5: Skipping outlier transformation (disabled in config)")

        # Step 6: Split and combine strategy
        print("\n>>> Step 7: Splitting and combining train/val/test")
        train_df, val_df, test_df, holdout_df = self._split_and_combine_data(
            main_df_new, holdout_df_new, combine_existing_data, has_new_classes
        )

        # Step 7: Tokenization (BERT tokenizer)
        print("\n>>> Step 8: Tokenizing train/val/test (holdout stays raw)")
        train_df = self._tokenize_dataframe(train_df)
        val_df = self._tokenize_dataframe(val_df)
        test_df = self._tokenize_dataframe(test_df)

        # Step 8: Save as parquet files
        print("\n>>> Step 9: Saving preprocessed data")
        output_paths = self._save_splits(train_df, val_df, test_df, holdout_df)

        # Add metadata to output
        output_paths["label_encoder_path"] = str(label_encoder_path)
        output_paths["num_labels"] = num_labels

        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Label encoder: {label_encoder_path}")
        print(f"Number of classes: {num_labels}")
        print(f"Output directory: {get_path('data.preprocessed')}")

        if holdout_df is not None:
            print("\n Holdout set saved with RAW text (not preprocessed)")
            print("    Use for realistic prediction/retraining simulation")

        return output_paths

    def _handle_label_encoding(
        self, df_new: pd.DataFrame, combine_existing_data: bool, has_new_classes: bool
    ) -> Tuple[pd.DataFrame, Path, int]:
        """
        Handle all label encoding logic in one place.

        Three scenarios: 1. Initial training: Fit new encoder on new data 2.
        Retraining, no new classes: Load existing encoder, encode new data 3.
        Retraining, new classes: Re-fit encoder on old+new prdtypecode

        Returns:
            df_new: DataFrame with 'labels' column added encoder_path: Path to
            label encoder file num_labels: Number of classes
        """
        models_dir = get_path("models.save_dir")
        models_dir.mkdir(parents=True, exist_ok=True)

        encoder_path = models_dir / "label_encoder.pkl"
        mapping_path = models_dir / "label_mappings.json"

        # Scenario 1: Initial training
        if not combine_existing_data:
            print("  Initial training: Fitting new encoder on new data")

            self.label_encoder = LabelEncoder()
            df_new["labels"] = self.label_encoder.fit_transform(df_new["prdtypecode"].values)

            print(f"    Classes: {len(self.label_encoder.classes_)}")

            # Save encoder
            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)

            # Save mappings
            self._save_label_mappings(mapping_path)

        # Scenario 2: Retraining, no new classes
        elif combine_existing_data and not has_new_classes:
            print("  Retraining (no new classes): Loading existing encoder")

            if not encoder_path.exists():
                raise FileNotFoundError(
                    f"Label encoder not found at {encoder_path}. Run initial training first!"
                )

            # Load existing encoder
            with open(encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)

            print(f"    Classes: {len(self.label_encoder.classes_)}")

            # Encode new data with existing encoder
            df_new["labels"] = self.label_encoder.transform(df_new["prdtypecode"].values)

        # Scenario 3: Retraining, new classes detected
        elif combine_existing_data and has_new_classes:
            print("  Retraining (new classes): Re-fitting encoder on ALL data")

            # Load old preprocessed data to get all prdtypecode values
            preprocessed_dir = get_path("data.preprocessed")
            train_old = pd.read_parquet(preprocessed_dir / "train.parquet")
            val_old = pd.read_parquet(preprocessed_dir / "val.parquet")
            test_old = pd.read_parquet(preprocessed_dir / "test.parquet")
            holdout_old = pd.read_parquet(preprocessed_dir / "holdout_raw.parquet")

            # Get all old categories
            old_categories = pd.concat(
                [
                    train_old["prdtypecode"],
                    val_old["prdtypecode"],
                    test_old["prdtypecode"],
                    holdout_old["prdtypecode"],
                ]
            ).unique()

            # Get new categories
            new_categories = df_new["prdtypecode"].unique()

            # Combine all categories
            all_categories = np.unique(np.concatenate([old_categories, new_categories]))

            print(f"    Old classes: {len(old_categories)}")
            print(f"    New data classes: {len(new_categories)}")
            print(f"    Total classes: {len(all_categories)}")

            # Fit new encoder on all categories
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(all_categories)

            # Encode new data
            df_new["labels"] = self.label_encoder.transform(df_new["prdtypecode"].values)

            # Save encoder
            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)

            # Save mappings
            self._save_label_mappings(mapping_path)

            print(f"    Re-fitted encoder with {len(self.label_encoder.classes_)} classes")
            print("    Old data will be re-encoded when loading splits")

        num_labels = len(self.label_encoder.classes_)

        return df_new, encoder_path, num_labels

    def _save_label_mappings(self, mapping_path: Path):
        """Save label mappings to JSON file."""
        id2label = {i: str(cat_id) for i, cat_id in enumerate(self.label_encoder.classes_)}
        label2id = {str(cat_id): i for i, cat_id in enumerate(self.label_encoder.classes_)}

        mappings = {
            "id2label": id2label,
            "label2id": label2id,
            "num_labels": len(self.label_encoder.classes_),
        }

        with open(mapping_path, "w") as f:
            json.dump(mappings, f, indent=2)

        print(f"    Saved mappings to: {mapping_path}")

    def _combine_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine 'designation' and 'description' into single 'text' column.

        Logic: - If description == designation: use only designation (avoid
        duplication) - Otherwise: concatenate with "; " separator

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

        print("  Combined designation + description into text column")

        return df

    def _split_holdout_early(self, df: pd.DataFrame) -> tuple:
        """
        Split holdout set BEFORE any preprocessing.

        This ensures holdout data remains "raw" and realistic for: - Testing
        prediction pipeline (text must go through full preprocessing) -
        Simulating retraining (new data arrives unprocessed)

        Returns:
            (main_df, holdout_df) - main_df: will be preprocessed and split into
            train/val/test - holdout_df: raw text, only combined
            designation+description
        """
        holdout_size = self.preproc_config["preprocessing"].get("holdout_size", 0.10)
        random_state = self.preproc_config["preprocessing"].get("random_state", 42)

        main_df, holdout_df = train_test_split(
            df, test_size=holdout_size, random_state=random_state, stratify=df["labels"]
        )

        print(
            f"Main data: {len(main_df)} ({len(main_df)/len(df) * 100:.1f}%) [will be preprocessed]"
        )
        print(
            f"Holdout: {len(holdout_df)} ({len(holdout_df) / len(df) * 100:.1f}%) [stays RAW]"
        )

        return main_df, holdout_df

    def _split_and_combine_data(
        self,
        df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        combine_existing_data: bool,
        has_new_classes: bool,
    ) -> tuple:
        """
        Split only NEW data, then combine with OLD splits (if
        combine_existing_data). If new classes are detected, the old splits will
        be re-encoded.

        Args:
            df: Preprocessed dataframe (new data, or combined but not yet split)
            combine_existing_data: Whether to combine with existing splits
            has_new_classes: If new classes were deteceted

        Returns:
            (train_df, val_df, test_df, holdout_df)
        """
        test_size = self.preproc_config["preprocessing"].get("test_size", 0.15)
        val_size = self.preproc_config["preprocessing"].get("val_size", 0.15)
        random_state = self.preproc_config["preprocessing"].get("random_state", 42)

        # Always split the current data (whether it's new-only or combined)
        # First split: (train+val) vs test
        train_val_df, test_df_new = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df["labels"]
        )

        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train_df_new, val_df_new = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df["labels"],
        )

        print("  Split new data:")
        print(f"    Train: {len(train_df_new)} ({len(train_df_new) / len(df) * 100:.1f}%)")
        print(f"    Val:   {len(val_df_new)} ({len(val_df_new) / len(df) * 100:.1f}%)")
        print(f"    Test:  {len(test_df_new)} ({len(test_df_new) / len(df) * 100:.1f}%)")

        # If combining, load old splits and combine
        if combine_existing_data:
            preprocessed_dir = get_path("data.preprocessed")
            train_file = preprocessed_dir / "train.parquet"
            val_file = preprocessed_dir / "val.parquet"
            test_file = preprocessed_dir / "test.parquet"
            holdout_file = preprocessed_dir / "holdout_raw.parquet"

            if train_file.exists() and val_file.exists() and test_file.exists():
                print("\n  Loading existing splits to combine:")

                # Load old splits (already preprocessed, but NOT tokenized yet)
                train_df_old = pd.read_parquet(train_file)
                val_df_old = pd.read_parquet(val_file)
                test_df_old = pd.read_parquet(test_file)
                holdout_df_old = pd.read_parquet(holdout_file)

                # Remove tokenization columns if they exist (will re-tokenize
                # later)
                for df_old in [train_df_old, val_df_old, test_df_old]:
                    if "input_ids" in df_old.columns:
                        df_old.drop(columns=["input_ids", "attention_mask"], inplace=True)

                # Re-encode old data if new classes detected
                if has_new_classes:
                    print("  Re-encoding old data with new encoder")
                    train_df_old["labels"] = self.label_encoder.transform(
                        train_df_old["prdtypecode"]
                    )
                    val_df_old["labels"] = self.label_encoder.transform(val_df_old["prdtypecode"])
                    test_df_old["labels"] = self.label_encoder.transform(test_df_old["prdtypecode"])
                    holdout_df_old["labels"] = self.label_encoder.transform(
                        holdout_df_old["prdtypecode"]
                    )

                print(f"    Old train: {len(train_df_old)}")
                print(f"    Old val:   {len(val_df_old)}")
                print(f"    Old test:  {len(test_df_old)}")

                # Combine each split separately
                train_df = pd.concat([train_df_old, train_df_new], ignore_index=True)
                val_df = pd.concat([val_df_old, val_df_new], ignore_index=True)
                test_df = pd.concat([test_df_old, test_df_new], ignore_index=True)
                holdout_df = pd.concat([holdout_df_old, holdout_df], ignore_index=True)

                print("\nCombined splits:")
                print(
                    f"Train: {len(train_df)} (old: {len(train_df_old)}, new: {len(train_df_new)})"
                )
                print(f"Val:   {len(val_df)} (old: {len(val_df_old)}, new: {len(val_df_new)})")
                print(
                    f"Test:  {len(test_df)} (old: {len(test_df_old)}, new: {len(test_df_new)})"
                )
            else:
                print("\n  No existing splits found - using new splits only")
                train_df = train_df_new
                val_df = val_df_new
                test_df = test_df_new
                holdout_df = holdout_df
        else:
            train_df = train_df_new
            val_df = val_df_new
            test_df = test_df_new
            holdout_df = holdout_df

        return train_df, val_df, test_df, holdout_df

    def _tokenize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tokenize all texts in dataframe using BERT tokenizer.

        Tokenization settings (same as training): - max_length: from config
        (default 128) - padding: "max_length" (all sequences same length) -
        truncation: True (cut long texts)

        Adds columns: - input_ids: token IDs for BERT - attention_mask: padding
        mask for BERT

        Returns:
            df with tokenized columns added
        """
        df = df.copy()

        # Batch tokenize for efficiency
        tokenized = self.tokenizer(
            df["text"].tolist(),
            max_length=self.preproc_config["preprocessing"]["max_length"],
            padding="max_length",
            truncation=True,
        )

        # Add tokenized columns
        df["input_ids"] = tokenized["input_ids"]
        df["attention_mask"] = tokenized["attention_mask"]

        print(f"  Tokenized {len(df)} samples")
        print(f"  Max length: {self.preproc_config['preprocessing']['max_length']}")

        return df

    def _save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        holdout_df: pd.DataFrame = None,
    ) -> dict:
        """
        Save train/val/test/holdout splits as parquet files.

        Train/Val/Test: Fully preprocessed and tokenized Holdout: Raw text with
        labels only (for realistic prediction testing)
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

        print(f"  Saved train -> {train_path} [preprocessed + tokenized]")
        print(f"  Saved val   -> {val_path} [preprocessed + tokenized]")
        print(f"  Saved test  -> {test_path} [preprocessed + tokenized]")

        result = {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "num_train": len(train_df),
            "num_val": len(val_df),
            "num_test": len(test_df),
        }

        # Save holdout (RAW text, only labels added)
        if holdout_df is not None:
            holdout_path = preprocessed_dir / "holdout_raw.parquet"
            # Save only: text (raw), designation, description, labels,
            # prdtypecode
            holdout_columns = ["designation", "description", "text", "labels", "prdtypecode"]
            holdout_df[holdout_columns].to_parquet(holdout_path, index=False)
            print(f"  Saved holdout -> {holdout_path} [RAW TEXT - not preprocessed]")

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
    output_paths = pipeline.prepare_training_data(
        df, combine_existing_data=False, save_holdout=True
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    for key, value in output_paths.items():
        print(f"{key}: {value}")
