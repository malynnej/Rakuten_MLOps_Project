#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Main wrapper function"""
    print("\n" + "=" * 60)
    print("MLFLOW EVALUATION WRAPPER")
    print("=" * 60 + "\n")

    import mlflow
    from src.evaluate_model.core.config import load_config, get_path

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ MLflow Tracking URI: {tracking_uri}")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten-evaluation")
    mlflow.set_experiment(experiment_name)
    print(f"✓ MLflow Experiment: {experiment_name}")

    # Timestamp
    ts_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_name = f"evaluate-{ts_utc}"

    git_sha = os.getenv("GIT_SHA", "unknown")
    git_branch = os.getenv("GIT_BRANCH", "unknown")

    params = load_config("params")
    train_params = params["training"]
    model_name = train_params.get("model_name", "bert-rakuten-final")

    print(f"✓ Model name: {model_name}")

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n✓ Started MLflow Run: {run.info.run_id}\n")

        # Set tags
        mlflow.set_tags({
            "service": "evaluate_api",
            "run_type": "evaluate",
            "timestamp_utc": ts_utc,
            "git_sha": git_sha,
            "git_branch": git_branch,
            "model_name": model_name,
            "wrapper_version": "1.0",
        })
        print("✓ Tags logged")

        print("\n" + "=" * 60)
        print("RUNNING ORIGINAL EVALUATION SCRIPT")
        print("=" * 60 + "\n")

        eval_script = Path(__file__).parent.parent / "src/evaluate_model/run_evaluate.py"
        cmd = [sys.executable, str(eval_script), "--model_name", model_name]

        result = subprocess.run(cmd, capture_output=False, text=True)

        if result.returncode != 0:
            print(f"\n Evaluation failed with return code {result.returncode}")
            mlflow.log_metric("evaluation_success", 0)
            sys.exit(result.returncode)

        print("\n✓ Evaluation completed successfully")
        mlflow.log_metric("evaluation_success", 1)

        metrics_file = Path("metrics/metrics.json")

        if metrics_file.exists():
            print(f"\n✓ Reading metrics from: {metrics_file}")
            
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key != "note":
                    mlflow.log_metric(key, value)
                    print(f"  ✓ {key}: {value}")
            
            print("✓ All metrics logged to MLflow")
        else:
            print(f"\n⚠ Warning: Metrics file not found at {metrics_file}")

        results_dir = get_path("results.evaluation")
        
        artifact_patterns = [
            "confusion_matrix.png",
            "classification_report.txt",
            "predictions.csv",
        ]

        for pattern in artifact_patterns:
            artifact_path = results_dir / pattern
            if artifact_path.exists():
                print(f"✓ Logging artifact: {pattern}")
                mlflow.log_artifact(str(artifact_path))

        print("\n" + "=" * 60)
        print("MLFLOW EVALUATION WRAPPER COMPLETE")
        print("=" * 60)
        print(f"Run ID: {run.info.run_id}")
        print(f"Experiment: {experiment_name}")
        print(f"Tracking URI: {tracking_uri}")
        
        if tracking_uri.startswith("http://mlflow"):
            print(f"View: {tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        else:
            print(f"View: {tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


if __name__ == "__main__":
    main()
