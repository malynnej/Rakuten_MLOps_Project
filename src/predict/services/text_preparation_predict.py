"""
Preprocessing pipeline for prediction/inference.

Applies the SAME transformations as training: - Text cleaning (HTML removal,
encoding fixes) - Outlier transformation (redundancy removal, summarization) -
Tokenization (BERT)

Handles: - Single text prediction (API requests) - Product field combination
(designation + description)
"""

import pickle

import pandas as pd
from core.config import get_path, load_config
from transformers import AutoTokenizer

from services.text_cleaning import TextCleaning
from services.text_outliers import TransformTextOutliers


class TextPreparationPredict:
    """
    Prediction preprocessing pipeline.

    Uses SAME preprocessing as training to ensure consistency: 1. Combine fields
    (designation + description) 2. Clean text (HTML, encoding) 3. Transform
    outliers (remove redundancy, summarize) 4. Tokenize (BERT)
    """

    def __init__(self, label_encoder_path: str = None):
        """
        Initialize prediction preprocessor.

        Args:
            label_encoder_path: Path to label encoder (for decoding predictions)
        """
        # Load configs
        self.preproc_config = load_config("params")

        model_path = get_path("models.final")

        # Initialize BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # Load label encoder if provided
        if label_encoder_path:
            with open(label_encoder_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print(f"Loaded label encoder with {len(self.label_encoder.classes_)} classes")
        else:
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

        print("TextPreparationPredict initialized successfully!")

    @classmethod
    def from_config(cls, config: dict, label_encoder_path: str = None):
        """
        Alternative constructor using config dict directly. Used by
        PredictionService.
        """
        return cls(label_encoder_path=label_encoder_path)

    def preprocess_text(self, text: str) -> dict:
        """
        Preprocess single text for prediction.

        Applies SAME transformations as training: 1. Clean text (HTML removal,
        encoding fixes) 2. Transform outliers (remove redundancy, summarize long
        texts) 3. Tokenize with BERT

        Args:
            text: Raw text string (can contain HTML, special characters)

        Returns:
            dict: {
                "input_ids": List[int], "attention_mask": List[int],
                "cleaned_text": str
            }

        Example:
            >>> prep = TextPreparationPredict()
            >>> result = prep.preprocess_text("<p>Nike shoes</p>")
            >>> result['cleaned_text']
            'Nike shoes'
            >>> len(result['input_ids'])
            128
        """
        # 1. Clean text (HTML, encoding)
        df_temp = pd.DataFrame({"text": [text]})
        df_cleaned, _ = self.text_cleaner.cleanTxt(df_temp, ["text"])
        cleaned_text = df_cleaned["text"].iloc[0]

        # Handle empty text
        if not cleaned_text or cleaned_text.strip() == "":
            cleaned_text = "[EMPTY]"

        # 2. Transform outliers (remove redundancy, summarize)
        if self.preproc_config["preprocessing"].get("transform_outliers", True):
            df_outlier = pd.DataFrame({"text": [cleaned_text]})
            df_transformed, _, _ = self.outlier_transformer.transform_outliers(df_outlier)
            cleaned_text = df_transformed["text"].iloc[0]

        # 3. Tokenize (BERT - same settings as training!)
        tokens = self.tokenizer(
            cleaned_text,
            max_length=self.preproc_config["preprocessing"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": tokens["input_ids"].tolist()[0],
            "attention_mask": tokens["attention_mask"].tolist()[0],
            "cleaned_text": cleaned_text,
        }

        return result

    def preprocess_product(self, designation: str, description: str = "") -> dict:
        """
        Preprocess product with designation and description.

        Combines fields using the same logic as training pipeline.

        Args:
            designation: Product title description: Product description
            (optional)

        Returns:
            dict: Same as preprocess_text()

        Example:
            >>> prep = TextPreparationPredict()
            >>> result = prep.preprocess_product(
            ...     "Nike Air Max",
            ...     "Running shoes with air cushioning"
            ... )
        """
        combined_text = self._combine_fields(designation, description)
        return self.preprocess_text(combined_text)

    def _combine_fields(self, designation: str, description: str) -> str:
        """
        Combine product fields into single text.

        Logic (SAME as training): - If description empty: use only designation -
        If description == designation: use only designation (avoid duplication)
        - Otherwise: "designation; description"
        """
        if not description or description.strip() == "":
            return designation

        if description == designation:
            return designation

        return f"{designation}; {description}"

    def decode_label(self, encoded_label: int) -> int:
        """
        Decode integer label back to original category ID.

        Args:
            encoded_label: Model prediction (0 to num_classes-1)

        Returns:
            Original prdtypecode (category ID)

        Example:
            >>> prep.decode_label(15)
            2403
        """
        if self.label_encoder is None:
            raise ValueError("Label encoder not loaded. Pass label_encoder_path to __init__().")

        return int(self.label_encoder.inverse_transform([encoded_label])[0])

    def get_num_labels(self) -> int:
        """Get number of classes (for model config)."""
        if self.label_encoder is None:
            raise ValueError("Label encoder not loaded")
        return len(self.label_encoder.classes_)


# Testing
if __name__ == "__main__":
    # Initialize
    models_dir = get_path("models.save_dir")
    encoder_path = models_dir / "label_encoder.pkl"

    prep = TextPreparationPredict(label_encoder_path=str(encoder_path))

    # Test single text
    print("\n" + "=" * 60)
    print("TEST: Single text preprocessing")
    print("=" * 60)

    test_text = "<p>Nike Air Max 90 - Premium running shoes</p>"
    result = prep.preprocess_text(test_text)

    print(f"Original text: {test_text}")
    print(f"Cleaned text: {result['cleaned_text']}")
    print(f"Input IDs length: {len(result['input_ids'])}")
    print(f"First 10 tokens: {result['input_ids'][:10]}")

    # Test product preprocessing
    print("\n" + "=" * 60)
    print("TEST: Product preprocessing")
    print("=" * 60)

    result2 = prep.preprocess_product(
        designation="Nike Air Max 90", description="Classic running shoes with visible Air unit"
    )
    print(f"Cleaned text: {result2['cleaned_text']}")

    # Test label decoding
    print("\n" + "=" * 60)
    print("TEST: Label decoding")
    print("=" * 60)

    test_label = 15
    category = prep.decode_label(test_label)
    print(f"Encoded label {test_label} → Category {category}")
