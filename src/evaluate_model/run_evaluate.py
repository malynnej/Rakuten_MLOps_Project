from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-rakuten-final")
    args = parser.parse_args()

    # DVC smoke test
    if os.getenv("DVC_FAST_EVAL") == "1":
        print("FAST EVAL MODE: writing dummy metrics only")
        Path("metrics").mkdir(exist_ok=True)

        metrics = {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "note": "dummy metrics for fast DVC pipeline validation",
        }

        with open("metrics/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return

    # Safety: prevent "real eval" on a fast/dummy model
    from core.config import get_path

    models_dir = get_path("models.save_dir")
    marker = models_dir / args.model_name / "FAST_TRAIN.txt"
    if marker.exists():
        raise RuntimeError(
            f"FAST model marker found at {marker}. "
            "Refusing real evaluation. Train a real model first or set DVC_FAST_EVAL=1."
        )

    from services.evaluate_text import ModelEvaluator

    evaluator = ModelEvaluator(model_name=args.model_name)
    results = evaluator.evaluate_dataset()

    Path("metrics").mkdir(exist_ok=True)

    metrics = {
        "accuracy": results.get("accuracy"),
        "f1_macro": results.get("f1_macro"),
        "f1_weighted": results.get("f1_weighted"),
    }

    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
