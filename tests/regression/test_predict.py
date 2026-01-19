"""Regression tests for TextPredictor baseline consistency."""
import json
from pathlib import Path
import pytest
from src.predict.predict_text import TextPredictor


# Load baselines from JSON file
def _load_baselines():
    baselines_file = Path(__file__).parent / "baselines_predict.json"
    with open(baselines_file, "r") as f:
        baseline_data = json.load(f)
    return baseline_data


BASELINE = _load_baselines()


def round_floats(obj, decimals=3):
    """Recursively round all floats in nested structures."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(round_floats(item, decimals) for item in obj)
    return obj


@pytest.fixture
def predictor():
    return TextPredictor(model_path="./models/bert-rakuten-final")


class TestPredictRegression:
    """Validate TextPredictor predictions match baseline"""
    
    @pytest.mark.parametrize("text,baseline", list(BASELINE["SIMPLE"].items()))
    def test_predict_simple(self, predictor, text, baseline):
        """Validate predict() matches baseline"""
        result = predictor.predict(text)
        assert round_floats(result, decimals=3) == round_floats(baseline, decimals=3)

    @pytest.mark.parametrize("text,baseline", list(BASELINE["WITH_PROBABILITIES"].items()))
    def test_predict_with_probabilities(self, predictor, text, baseline):
        """Validate predict() with return_probabilities=True"""
        result = predictor.predict(text, return_probabilities=True)
        assert round_floats(result, decimals=3) == round_floats(baseline, decimals=3)

    @pytest.mark.parametrize("text,baseline", list(BASELINE["BATCH_PREDICTION"].items()))
    def test_predict_batch(self, predictor, text, baseline):
        """Validate predict_batch() matches baseline"""
        result = predictor.predict(text)
        assert round_floats(result, decimals=3) == round_floats(baseline, decimals=3)
