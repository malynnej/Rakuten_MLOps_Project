import os
from datetime import datetime, timezone

import mlflow


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "rakuten-smoke")
    mlflow.set_experiment(exp_name)

    git_sha = os.getenv("GIT_SHA", "unknown")
    git_branch = os.getenv("GIT_BRANCH", "unknown")
    service = os.getenv("SERVICE_NAME", "mlflow_smoke")
    run_type = os.getenv("RUN_TYPE", "smoke")

    ts_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with mlflow.start_run(run_name=f"smoke-{ts_utc}"):
        mlflow.set_tags(
            {
                "service": service,
                "run_type": run_type,
                "timestamp_utc": ts_utc,
                "git_sha": git_sha,
                "git_branch": git_branch,
            }
        )
        mlflow.log_metric("smoke_ok", 1.0)
        mlflow.log_param("tracking_uri", tracking_uri)

    print("OK: logged smoke run to", tracking_uri)


if __name__ == "__main__":
    main()
