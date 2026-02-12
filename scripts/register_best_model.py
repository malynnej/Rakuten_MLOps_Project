#!/usr/bin/env python3
"""
Register best model to MLflow Model Registry
"""

import os
import sys
from pathlib import Path

from mlflow.tracking import MlflowClient

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def register_best_model(experiment_name: str, metric_name: str = "accuracy"):
    """
    Find best run and register model to registry

    Args:
        experiment_name: Name of experiment to search
        metric_name: Metric to optimize (default: accuracy)
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found")
        return None

    print(f"✓ Found experiment: {experiment_name} (ID={experiment.experiment_id})")

    # Search for best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )

    if not runs:
        print(f"No runs found in experiment {experiment_name}")
        return None

    best_run = runs[0]
    run_id = best_run.info.run_id
    best_metric = best_run.data.metrics.get(metric_name, 0)

    print(f"✓ Best run: {run_id}")
    print(f"  {metric_name}: {best_metric}")

    # Register model
    model_name = "rakuten-text-classifier"
    model_uri = f"runs:/{run_id}/model"

    try:
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

        print(f"✓ Registered model: {model_name}")
        print(f"  Version: {model_version.version}")
        print(f"  Run ID: {run_id}")

        # Set alias for latest
        client.set_registered_model_alias(
            name=model_name, alias="champion", version=model_version.version
        )
        print(f"  Alias 'champion' set to version {model_version.version}")

        return model_version

    except Exception as e:
        print(f"⚠ Model registration: {e}")
        print("  Note: Model registry might not be fully supported on this MLflow instance")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", default="rakuten-evaluation", help="Experiment name to search"
    )
    parser.add_argument("--metric", default="accuracy", help="Metric to optimize")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MODEL REGISTRY")
    print("=" * 60 + "\n")

    register_best_model(args.experiment, args.metric)
