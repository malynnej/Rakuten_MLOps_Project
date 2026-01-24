"""
Training API - Background training with status tracking
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.train_model_text import train_bert_model

app = FastAPI(title="Rakuten ML Training API")


# ============================================
# GLOBAL STATE
# ============================================

# Training status tracker
training_status = {
    "is_training": False,
    "current_epoch": None,
    "status": "idle",
    "metrics": None,
    "last_training": None
}


# ============================================
# REQUEST MODELS
# ============================================

class TrainRequest(BaseModel):
    retrain: bool = False
    model_name: str = "bert-rakuten-final"


# ============================================
# TRAINING ENDPOINT
# ============================================

def run_training(retrain: bool, model_name: str):
    """Background task for training"""
    global training_status
    
    try:
        # Update status
        training_status["is_training"] = True
        training_status["status"] = "training"
        training_status["last_training"] = datetime.now().isoformat()
        
        print(f"\n{'='*60}")
        print(f"BACKGROUND TRAINING STARTED")
        print(f"{'='*60}\n")
        print(f"Mode: {'Retraining' if retrain else 'Initial training'}")
        print(f"Model: {model_name}")
        
        # Train model
        trainer, metrics = train_bert_model(retrain=retrain, model_name=model_name)
        
        # Update status with success
        training_status["is_training"] = False
        training_status["status"] = "completed"
        training_status["metrics"] = {
            "accuracy": metrics.get("test_accuracy"),
            "train_loss": metrics.get("final_train_loss"),
            "test_loss": metrics.get("test_loss"),
            "num_labels": metrics.get("num_labels"),
            "trainable_params": metrics.get("trainable_params"),
            "training_duration": metrics.get("training_duration_seconds")
        }
        
        print(f"\n✅ Background training completed!")
        print(f"   Test accuracy: {metrics.get('test_accuracy', 'N/A')}")
        print(f"   Model saved to: {metrics.get('model_path', 'N/A')}")
        
    except Exception as e:
        # Update status with error
        training_status["is_training"] = False
        training_status["status"] = f"failed: {str(e)}"
        training_status["metrics"] = None
        
        print(f"\n❌ Background training failed: {e}")


@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    Start model training.
    Runs in background to avoid timeout.
    
    Example:
        POST /train
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
            detail="Training already in progress. Check /status for details."
        )
    
    # Start training in background
    background_tasks.add_task(run_training, request.retrain, request.model_name)
    
    return {
        "status": "training_started",
        "message": f"Training job submitted. Check /status for progress.",
        "retrain": request.retrain,
        "model_name": request.model_name,
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# STATUS & INFO ENDPOINTS
# ============================================

@app.get("/status")
async def get_status():
    """
    Get current training status.
    
    Returns:
        Current training status and metrics if completed
    """
    return {
        "training_status": training_status,
        "timestamp": datetime.now().isoformat()
    }


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
        "timestamp": datetime.now().isoformat()
    }


@app.get("/results/latest")
async def get_latest_results():
    """
    Get results from most recent training.
    
    Returns:
        Latest training metrics or 404 if none available
    """
    if training_status["metrics"] is None:
        raise HTTPException(
            status_code=404,
            detail="No training results available"
        )
    
    return {
        "status": "success",
        "metrics": training_status["metrics"],
        "timestamp": training_status["last_training"]
    }


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
async def root():
    """API root with usage information"""
    return {
        "service": "Rakuten ML Training API",
        "version": "1.0.0",
        "endpoints": {
            "train": "POST /train - Start model training",
            "status": "GET /status - Get training status",
            "health": "GET /health - Health check",
            "latest_results": "GET /results/latest - Latest training metrics"
        },
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8002))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
