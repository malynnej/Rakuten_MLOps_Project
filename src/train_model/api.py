# src/train_model/api.py
"""
Training API - Background training with status tracking

Handles:
1. Model training (initial and retraining)
2. Training status monitoring
3. Results retrieval
"""

import os
import traceback
from datetime import datetime

from core.config import get_path
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from services.train_model_text import train_bert_model

API_ROOT_PATH = "/train"

app = FastAPI(
    title="Training Service - Rakuten MLOps",
    root_path=API_ROOT_PATH,
)

instrumentator = Instrumentator().instrument(app)


@app.on_event("startup")
async def startup_event():
    """Initialize predictor when API starts"""
    instrumentator.expose(app)


# Workaround to make docs available behind proxy AND locally
@app.get(f"{API_ROOT_PATH}/openapi.json", include_in_schema=False)
async def get_docs():
    return RedirectResponse(url="/openapi.json")


# Global state
training_status = {
    "is_training": False,
    "current_epoch": None,
    "status": "idle",
    "metrics": None,
    "last_training": None,
    "error_details": None,
    "progress": 0,
}


# Request models
class TrainRequest(BaseModel):
    retrain: bool = False
    model_name: str = "bert-rakuten-final"


# Background training task
def run_training(retrain: bool, model_name: str):
    """Background task for training with enhanced error handling"""
    global training_status

    try:
        # Update status
        training_status["is_training"] = True
        training_status["status"] = "training"
        training_status["last_training"] = datetime.now().isoformat()
        training_status["error_details"] = None
        training_status["progress"] = 0

        print(f"\n{'=' * 60}")
        print("BACKGROUND TRAINING STARTED")
        print(f"{'=' * 60}\n")
        print(f"Mode: {'Retraining' if retrain else 'Initial training'}")
        print(f"Model: {model_name}")

        # Train model
        trainer, metadata = train_bert_model(retrain=retrain, model_name=model_name)

        # Update status with success
        training_status["is_training"] = False
        training_status["status"] = "completed"
        training_status["progress"] = 100
        training_status["metrics"] = {
            "test_accuracy": metadata.get("test_accuracy"),
            "train_loss": metadata.get("final_train_loss"),
            "test_loss": metadata.get("test_loss"),
            "num_labels": metadata.get("num_labels"),
            "trainable_params": metadata.get("trainable_params"),
            "total_params": metadata.get("total_params"),
            "training_duration_seconds": metadata.get("training_duration_seconds"),
            "train_samples": metadata.get("train_samples"),
            "val_samples": metadata.get("val_samples"),
            "test_samples": metadata.get("test_samples"),
            "device": metadata.get("device"),
            "mode": metadata.get("mode"),
            "base_model": metadata.get("base_model"),
            "model_path": str(get_path("models.save_dir") / model_name),
            "mlflow_run_id": metadata.get("mlflow_run_id"),
        }

        print("\n Background training completed successfully!")
        print(f"   Test accuracy: {metadata.get('test_accuracy', 'N/A'):.4f}")
        print(f"   Duration: {metadata.get('training_duration_seconds', 'N/A'):.1f}s")

    except FileNotFoundError as e:
        training_status["is_training"] = False
        training_status["status"] = "failed"
        training_status["progress"] = 0
        training_status["metrics"] = None
        training_status["error_details"] = {
            "error_type": "FileNotFoundError",
            "message": str(e),
            "suggestion": "Run data preprocessing first: POST http://localhost:8001/preprocess/from-raw",
        }
        print(f"\n Training failed - File not found: {e}")

    except Exception as e:
        training_status["is_training"] = False
        training_status["status"] = "failed"
        training_status["progress"] = 0
        training_status["metrics"] = None
        training_status["error_details"] = {
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"\n Training failed: {e}")
        traceback.print_exc()


# Training endpoint
@app.post("/train_model")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start model training in background.

    Args:
        request: TrainRequest with retrain flag and model_name

    Returns:
        Training job status

    Example:
        POST /train_model
        {
            "retrain": false,
            "model_name": "bert-rakuten-final"
        }
    """
    global training_status

    # Check if training already in progress
    if training_status["is_training"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Training already in progress",
                "current_status": training_status["status"],
                "started_at": training_status["last_training"],
            },
        )

    # Validate prerequisites
    preprocessed_dir = get_path("data.preprocessed")
    train_file = preprocessed_dir / "train.parquet"

    if not train_file.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Training data not found",
                "missing_file": str(train_file),
                "suggestion": "Run data preprocessing first: POST http://localhost:8001/preprocess/from-raw",
            },
        )

    # Check if retraining but model doesn't exist
    if request.retrain:
        models_dir = get_path("models.save_dir")
        model_path = models_dir / request.model_name

        if not model_path.exists():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Cannot retrain - model not found",
                    "model_path": str(model_path),
                    "suggestion": "Run initial training first (set retrain=false)",
                },
            )

    # Start training in background
    background_tasks.add_task(run_training, request.retrain, request.model_name)

    return {
        "status": "training_started",
        "message": "Training job submitted. Monitor progress at /status",
        "retrain": request.retrain,
        "model_name": request.model_name,
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "status": "/status",
            "results": "/results/latest",
            "cancel": "/cancel (not implemented)",
        },
    }


# Status endpoint
@app.get("/status")
async def get_status():
    """
    Get current training status with detailed metrics.

    Returns:
        Current training status, progress, and metrics if completed
    """
    return {"training_status": training_status, "timestamp": datetime.now().isoformat()}


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        API health status and training state
    """
    return {
        "status": "healthy",
        "service": "training-service",
        "is_training": training_status["is_training"],
        "last_training": training_status["last_training"],
        "timestamp": datetime.now().isoformat(),
    }


# Latest results endpoint
@app.get("/results/latest")
async def get_latest_results():
    """
    Get results from most recent training.

    Returns:
        Latest training metrics or 404 if none available
    """
    if training_status["metrics"] is None:
        raise HTTPException(
            status_code=404, detail="No training results available. Train a model first."
        )

    return {
        "status": "success",
        "metrics": training_status["metrics"],
        "training_completed": training_status["last_training"],
        "timestamp": datetime.now().isoformat(),
    }


# Prerequisites check endpoint
@app.get("/prerequisites")
async def check_prerequisites():
    """
    Check if all prerequisites for training are met.

    Returns:
        Status of required data files and models
    """
    preprocessed_dir = get_path("data.preprocessed")
    models_dir = get_path("models.save_dir")

    checks = {
        "data_preprocessed": {
            "train": (preprocessed_dir / "train.parquet").exists(),
            "val": (preprocessed_dir / "val.parquet").exists(),
            "test": (preprocessed_dir / "test.parquet").exists(),
        },
        "label_encoder": {
            "encoder": (models_dir / "label_encoder.pkl").exists(),
            "mappings": (models_dir / "label_mappings.json").exists(),
        },
        "models": {},
    }

    # Check existing models
    if models_dir.exists():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                checks["models"][model_dir.name] = {"exists": True, "path": str(model_dir)}

    all_ready = all(checks["data_preprocessed"].values()) and all(checks["label_encoder"].values())

    return {
        "ready_for_training": all_ready,
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
    }


# Root endpoint
@app.get("/")
async def root():
    """API root with usage information"""
    return {
        "service": "Rakuten ML Training API",
        "version": "1.0.0",
        "description": "Background training service for BERT text classification",
        "endpoints": "endpoints for accessing service",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }


# Run server
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
