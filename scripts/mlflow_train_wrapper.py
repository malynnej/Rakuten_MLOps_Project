#!/usr/bin/env python3
"""
MLflow Training Wrapper

Wraps the existing train service without modifying it.
"""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("\n" + "=" * 60)
    print("MLFLOW TRAINING WRAPPER")
    print("=" * 60 + "\n")

    import mlflow
    from src.train_model.core.config import get_path, load_config

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ MLflow Tracking URI: {tracking_uri}")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten-training")
    mlflow.set_experiment(experiment_name)
    print(f"✓ MLflow Experiment: {experiment_name}")

    ts_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_name = f"train-{ts_utc}"

    git_sha = os.getenv("GIT_SHA", "unknown")
    git_branch = os.getenv("GIT_BRANCH", "unknown")

    params = load_config("params")
    train_params = params["training"]
    model_name = train_params.get("model_name", "bert-rakuten-final")

    print(f"✓ Model name: {model_name}")

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n✓ Started MLflow Run: {run.info.run_id}\n")

        mlflow.set_tags(
            {
                "service": "train_api",
                "run_type": "train",
                "timestamp_utc": ts_utc,
                "git_sha": git_sha,
                "git_branch": git_branch,
                "model_name": model_name,
                "wrapper_version": "1.0",
            }
        )
        print("✓ Tags logged")

        mlflow.log_params(
            {
                "num_train_epochs": train_params["num_train_epochs"],
                "batch_size": train_params["per_device_train_batch_size"],
                "learning_rate": train_params["learning_rate"],
                "weight_decay": train_params["weight_decay"],
                "warmup_ratio": train_params["warmup_ratio"],
                "llrd_enabled": train_params["llrd"]["enabled"],
                "llrd_decay_factor": train_params["llrd"]["lr_decay_factor"],
                "freeze_embeddings": train_params["freeze_embeddings"],
                "early_stopping_patience": train_params["early_stopping"]["patience"],
                "fp16": train_params["fp16"],
            }
        )
        print("✓ Parameters logged")

        print("\n" + "=" * 60)
        print("RUNNING ORIGINAL TRAINING SCRIPT")
        print("=" * 60 + "\n")

        train_script = Path(__file__).parent.parent / "src/train_model/run_train.py"
        cmd = [sys.executable, str(train_script), "--model_name", model_name]

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"\nTraining failed with return code {result.returncode}")
            mlflow.log_metric("training_success", 0)
            sys.exit(result.returncode)

        print("\n✓ Training completed successfully")
        mlflow.log_metric("training_success", 1)

        models_dir = get_path("models.save_dir")
        model_path = models_dir / model_name

        if model_path.exists():
            print(f"\n✓ Model found at: {model_path}")
            if os.getenv("MLFLOW_LOG_MODEL", "false").lower() == "true":
                print("  Logging model to MLflow (this may take a while)...")
                mlflow.log_artifacts(str(model_path), artifact_path="model")
                print("  ✓ Model artifacts logged")
            else:
                print("  ℹ Model not logged (set MLFLOW_LOG_MODEL=true to enable)")
                mlflow.log_param("model_path", str(model_path))

        print("\n" + "=" * 60)
        print("MLFLOW TRAINING WRAPPER COMPLETE")
        print("=" * 60)
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Tracking URI: {tracking_uri}")


if __name__ == "__main__":
    main()
