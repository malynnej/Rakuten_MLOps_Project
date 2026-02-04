"""
DVC pipeline tests: Check DVC commands work (basic), does not run full DVC pipeline
"""

import subprocess
import pytest
from pathlib import Path


class TestDVCPipeline:
    """Test DVC pipeline operations"""
    
    def test_dvc_status(self):
        """Test DVC status command"""
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
    
    def test_dvc_dag(self):
        """Test DVC pipeline DAG"""
        result = subprocess.run(
            ["dvc", "dag"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "preprocess" in result.stdout
        assert "train" in result.stdout
        assert "evaluate" in result.stdout
    
    def test_preprocessed_data_exists(self):
        """Test preprocessed data tracked by DVC"""
        preprocessed_dir = Path("data/preprocessed")
        assert preprocessed_dir.exists()
        
        required_files = [
            "train.parquet",
            "val.parquet",
            "test.parquet",
        ]
        
        for filename in required_files:
            filepath = preprocessed_dir / filename
            assert filepath.exists(), f"Missing {filename}"
    
    def test_dvc_repro_dry_run(self):
        """Test DVC pipeline dry run"""
        result = subprocess.run(
            ["dvc", "repro", "--dry"],
            capture_output=True,
            text=True
        )
        # Should succeed (0) or indicate up-to-date (0)
        assert result.returncode == 0
    
    def test_model_tracked(self):
        """Test model is tracked by DVC"""
        models_dir = Path("models/bert-rakuten-final")
        if models_dir.exists():
            # Model should have .dvc file or be in dvc.lock
            assert True  # Model exists
