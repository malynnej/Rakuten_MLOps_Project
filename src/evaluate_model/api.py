# src/evaluation/api.py
"""
Evaluation API - Model evaluation service

Handles:
- Model evaluation on test data
- Classification reports
- Confusion matrices
- Confidence analysis
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# Set non-interactive backend BEFORE importing pyplot
import matplotlib
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

matplotlib.use("Agg")  # Use non-interactive backend for background tasks

from core.config import get_path
from services.evaluate_text import ModelEvaluator

app = FastAPI(title="Rakuten ML Evaluation API")

# Global state
evaluator = None
evaluation_status = {
    "is_evaluating": False,
    "status": "idle",
    "last_evaluation": None,
    "results": None,
    "error_details": None,
}


@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    global evaluator

    print("=" * 60)
    print("INITIALIZING EVALUATION SERVICE")
    print("=" * 60)

    try:
        # Get model info from environment or config
        models_dir = get_path("models.save_dir")
        model_name = os.environ.get("MODEL_NAME", "bert-rakuten-final")
        model_path = models_dir / model_name

        print(f"\nLoading model: {model_path}")

        evaluator = ModelEvaluator(model_path=str(model_path))

        print("\nEvaluation service ready!")
        print(f"  Model: {model_name}")
        print(f"  Device: {evaluator.device}")
        print(f"  Classes: {evaluator.num_labels}")

    except Exception as e:
        print(f"\nFailed to load model: {e}")
        evaluator = None
        raise


# Request models
class EvaluateRequest(BaseModel):
    test_path: Optional[str] = None
    output_dir: Optional[str] = None
    batch_size: int = 32
    model_name: str = "bert-rakuten-final"


# Root endpoint
@app.get("/")
async def root():
    """API root with usage information"""
    return {
        "service": "Rakuten ML Evaluation API",
        "version": "1.0.0",
        "description": "Model evaluation and performance analysis",
        "endpoints": {
            "evaluate": "POST /evaluate - Run model evaluation",
            "status": "GET /status - Evaluation status",
            "model_info": "GET /model/info - Model information",
            "results": "GET /results/latest - Latest evaluation results",
            "health": "GET /health - Health check",
        },
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }


# Evaluation endpoint
@app.post("/evaluate")
async def evaluate_model(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """
    Evaluate model on test dataset.
    Runs in background to avoid timeout.

    Args:
        test_path: Path to test.parquet (optional, uses config default)
        output_dir: Output directory for results (optional, uses config default)
        batch_size: Batch size for evaluation (default: 32)
        model_name: Model to evaluate (default: bert-rakuten-final)

    Returns:
        Evaluation job status

    Example:
        POST /evaluate
        {
            "test_path": null,
            "output_dir": "./results/evaluation",
            "batch_size": 32,
            "model_name": "bert-rakuten-final"
        }

        Response:
        {
            "status": "evaluation_started",
            "message": "Evaluation job submitted",
            "model_name": "bert-rakuten-final",
            "output_dir": "./results/evaluation/bert-rakuten-final",
            "timestamp": "2026-01-28T01:05:00"
        }
    """
    global evaluation_status

    # Check if evaluation already running
    if evaluation_status["is_evaluating"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Evaluation already in progress",
                "current_status": evaluation_status["status"],
                "started_at": evaluation_status["last_evaluation"],
            },
        )

    # Check if evaluator loaded
    if evaluator is None:
        raise HTTPException(
            status_code=503, detail="Evaluator not initialized. Check /health endpoint."
        )

    # Validate test data exists
    if request.test_path:
        test_path = Path(request.test_path)
    else:
        preprocessed_dir = get_path("data.preprocessed")
        test_path = preprocessed_dir / "test.parquet"

    if not test_path.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Test data not found",
                "test_path": str(test_path),
                "suggestion": "Run data preprocessing first: POST http://localhost:8001/preprocess/from-raw",
            },
        )

    # Background evaluation job
    def evaluation_job():
        global evaluation_status

        try:
            # Update status
            evaluation_status["is_evaluating"] = True
            evaluation_status["status"] = "evaluating"
            evaluation_status["last_evaluation"] = datetime.now().isoformat()
            evaluation_status["error_details"] = None

            print(f"\n{'=' * 60}")
            print("BACKGROUND EVALUATION STARTED")
            print(f"{'=' * 60}\n")
            print(f"Model: {request.model_name}")
            print(f"Test path: {test_path}")
            print(f"Batch size: {request.batch_size}")

            # Load test data
            test_dataset, _ = evaluator.load_test_data(test_path)

            # Run evaluation
            results = evaluator.evaluate_dataset(
                dataset=test_dataset,
                batch_size=request.batch_size,
                output_dir=request.output_dir,
                model_name=request.model_name,
            )

            # Update status with success
            evaluation_status["is_evaluating"] = False
            evaluation_status["status"] = "completed"
            evaluation_status["results"] = {
                "accuracy": results["accuracy"],
                "timestamp": results["timestamp"],
                "dataset_size": results["dataset_size"],
                "output_files": results["output_files"],
            }

            print("\n Background evaluation completed!")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   Results: {results['output_files']}")

        except Exception as e:
            # Update status with error
            evaluation_status["is_evaluating"] = False
            evaluation_status["status"] = "failed"
            evaluation_status["results"] = None
            evaluation_status["error_details"] = {"error_type": type(e).__name__, "message": str(e)}

            print(f"\n Background evaluation failed: {e}")

    # Determine output directory
    if request.output_dir:
        output_dir = request.output_dir
    else:
        output_dir = str(get_path("results.evaluation") / request.model_name)

    # Start background task
    background_tasks.add_task(evaluation_job)

    return {
        "status": "evaluation_started",
        "message": "Evaluation job submitted. Monitor progress at /status",
        "model_name": request.model_name,
        "output_dir": output_dir,
        "timestamp": datetime.now().isoformat(),
    }


# Status endpoint
@app.get("/status")
async def get_status():
    """
    Get current evaluation status.

    Returns:
        Evaluation status with results if completed
    """
    return {"evaluation_status": evaluation_status, "timestamp": datetime.now().isoformat()}


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "evaluation-service",
        "model_loaded": evaluator is not None,
        "is_evaluating": evaluation_status["is_evaluating"],
        "timestamp": datetime.now().isoformat(),
    }


# Model info
@app.get("/model/info")
async def model_info():
    """
    Get information about loaded model.

    Returns:
        Model configuration and metadata
    """
    if evaluator is None:
        raise HTTPException(
            status_code=503, detail="Evaluator not initialized. Check /health endpoint."
        )

    return {
        "model_path": str(evaluator.model_path),
        "num_labels": evaluator.num_labels,
        "device": str(evaluator.device),
        "label_encoder_classes": len(evaluator.le.classes_),
        "id2label_sample": dict(list(evaluator.id2label.items())[:5]),
        "timestamp": datetime.now().isoformat(),
    }


# Latest results
@app.get("/results/latest")
async def get_latest_results():
    """
    Get results from most recent evaluation.

    Returns:
        Latest evaluation results or 404 if none available
    """
    if evaluation_status["results"] is None:
        raise HTTPException(
            status_code=404, detail="No evaluation results available. Run evaluation first."
        )

    return {
        "status": "success",
        "results": evaluation_status["results"],
        "evaluated_at": evaluation_status["last_evaluation"],
        "timestamp": datetime.now().isoformat(),
    }


# Run server
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8004))

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
