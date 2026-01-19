"""Generate baseline predictions for test_predict_regression.py.

Generates baselines in JSON format and saves to predict_regression_baselines.json

Usage:
    python -m tests.regression.capture_predict_regression_baselines
"""

import json
from pathlib import Path
from src.predict.predict_text import TextPredictor

TEST_TEXTS = [
    "Solar powered garden lamp with automatic night light sensor",
    "Portable handheld console gaming device with 400 classic games",
    "Professional carpet cleaning equipment rental",
    "Leather office chair with ergonomic lumbar support",
    "Zombie action figure collectible model"
]


def capture_simple_predictions():
    """Capture predict() without probabilities"""
    predictor = TextPredictor(model_path="./models/bert-rakuten-final")
    result = {}
    for text in TEST_TEXTS:
        result[text] = predictor.predict(text)
        print(f"✓ {text[:50]}...")
    return result


def capture_predictions_with_probabilities():
    """Capture predict() with return_probabilities=True"""
    predictor = TextPredictor(model_path="./models/bert-rakuten-final")
    result = {}
    for text in TEST_TEXTS:
        result[text] = predictor.predict(text, return_probabilities=True)
        print(f"✓ {text[:50]}...")
    return result

def capture_batch_prediction():
    """Capture predict() for batch input"""
    predictor = TextPredictor(model_path="./models/bert-rakuten-final")
    result = {}
    batch_result = predictor.predict(TEST_TEXTS)
    for text, prediction in zip(TEST_TEXTS, batch_result):
        result[text] = prediction
        print(f"✓ {text[:50]}...")
    return result


if __name__ == "__main__":
    print("Capturing BASELINE_SIMPLE...\n")
    simple = capture_simple_predictions()
    
    print("\nCapturing BASELINE_WITH_PROBABILITIES...\n")
    with_probs = capture_predictions_with_probabilities()

    print("\nCapturing BASELINE_BATCH_PREDICTION...\n")
    batch_preds = capture_batch_prediction()
    
    # Save to JSON file
    baselines = {
        "SIMPLE": simple,
        "WITH_PROBABILITIES": with_probs,
        "BATCH_PREDICTION": batch_preds
    }
    
    baselines_file = Path(__file__).parent / "baselines_predict.json"
    with open(baselines_file, "w") as f:
        json.dump(baselines, f, indent=2)
    
    print(f"\n✓ Baselines saved to {baselines_file}")
