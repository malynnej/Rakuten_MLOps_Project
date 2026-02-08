from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="bert-rakuten-final")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    # Smoke Test
    import os

    if os.getenv("DVC_FAST_TRAIN") == "1":
        print("FAST TRAIN MODE: skipping real training")

        from core.config import get_path

        models_dir = get_path("models.save_dir")
        model_path = models_dir / args.model_name
        model_path.mkdir(parents=True, exist_ok=True)

        (model_path / "FAST_TRAIN.txt").write_text(
            "Dummy model created for fast DVC pipeline validation\n"
        )

        return

    from services.train_model_text import train_bert_model

    train_bert_model(retrain=args.retrain, model_name=args.model_name)


if __name__ == "__main__":
    main()
