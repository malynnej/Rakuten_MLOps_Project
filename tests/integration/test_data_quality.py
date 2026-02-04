"""
Data quality and integrity tests
"""

import pandas as pd
import pytest
from pathlib import Path


class TestDataQuality:
    """Test data quality throughout pipeline"""
    
    def test_preprocessed_data_not_empty(self):
        """Test preprocessed files contain data"""
        preprocessed_dir = Path("data/preprocessed")
        
        for filename in ["train.parquet", "val.parquet", "test.parquet"]:
            filepath = preprocessed_dir / filename
            
            if filepath.exists():
                df = pd.read_parquet(filepath)
                assert len(df) > 0, f"{filename} is empty"
    
    def test_preprocessed_data_columns(self):
        """Test preprocessed data has required columns"""
        preprocessed_dir = Path("data/preprocessed")
        train_path = preprocessed_dir / "train.parquet"
        
        if train_path.exists():
            df = pd.read_parquet(train_path)
            
            required_columns = [
                "text",
                "input_ids",
                "attention_mask",
                "labels",
                "prdtypecode"
            ]
            
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
    
    def test_no_null_values(self):
        """Test preprocessed data has no null values"""
        preprocessed_dir = Path("data/preprocessed")
        train_path = preprocessed_dir / "train.parquet"
        
        if train_path.exists():
            df = pd.read_parquet(train_path)
            
            critical_columns = ["text", "labels", "input_ids"]
            for col in critical_columns:
                null_count = df[col].isnull().sum()
                assert null_count == 0, (
                    f"Column '{col}' has {null_count} null values"
                )
    
    def test_label_distribution(self):
        """Test label distribution is reasonable"""
        preprocessed_dir = Path("data/preprocessed")
        train_path = preprocessed_dir / "train.parquet"
        
        if train_path.exists():
            df = pd.read_parquet(train_path)
            
            # Check we have multiple classes
            num_classes = df["labels"].nunique()
            assert num_classes > 1, "Only one class found"
            
            # Check no class dominates too much (>95%)
            max_class_pct = df["labels"].value_counts(normalize=True).max()
            assert max_class_pct < 0.95, (
                f"One class dominates: {max_class_pct:.1%}"
            )
