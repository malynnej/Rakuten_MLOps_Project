"""
Evaluation API 
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.evaluate_text import ModelEvaluator
from core.config import get_path

app = FastAPI(title="Rakuten ML Evaluate API")


# ============================================
# GLOBAL STATE
# ============================================

# Global evaluator (loaded once at startup)
evaluator = None

# Model configuration from environment or defaults
models_dir = os.environ.get("MODELS_DIR", str(get_path("models.save_dir")))
model_name = os.environ.get("MODEL_NAME", "bert-rakuten-final")
current_model_path = os.path.join(models_dir, model_name)


@app.on_event("startup")
async def load_model():
    """Load model when API starts"""
    global evaluator
    try:
        evaluator = ModelEvaluator(model_path=current_model_path)
        print(f" Model loaded at startup: {current_model_path}")
        print(f"   Device: {evaluator.device}")
        print(f"   Num labels: {evaluator.num_labels}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        evaluator = None


# ============================================
# REQUEST MODELS
# ============================================

class EvaluateRequest(BaseModel):
    test_path: Optional[str] = None  # Path to test.parquet (default: from config)
    output_dir: Optional[str] = None  # Output directory (default: from config)
    batch_size: int = 32


# ============================================
# EVALUATION ENDPOINT
# ============================================

@app.post("/evaluate")
async def evaluate_model(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """
    Evaluate model on test dataset.
    Runs in background to avoid timeout.
    
    Example:
        POST /evaluate
        {
            "test_path": null,
            "output_dir": "./results/evaluation",
            "batch_size": 32
        }
    """
    if evaluator is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check API startup logs."
        )
    
    def evaluation_job():
        try:
            print(f"\n{'='*60}")
            print("BACKGROUND EVALUATION STARTED")
            print(f"{'='*60}\n")
            
            # Load test data
            if request.test_path:
                test_dataset, _ = evaluator.load_test_data(Path(request.test_path))
            else:
                test_dataset, _ = evaluator.load_test_data()
            
            # Run evaluation
            results = evaluator.evaluate_dataset(
                dataset=test_dataset,
                batch_size=request.batch_size,
                output_dir=request.output_dir
            )
            
            print(f"\n Evaluation complete!")
            print(f"   Accuracy: {results['accuracy']:.4f}")
            print(f"   Results saved to: {request.output_dir or 'default'}")
            
        except Exception as e:
            print(f" Evaluation failed: {e}")
    
    # Start background task
    background_tasks.add_task(evaluation_job)
    
    return {
        "status": "evaluation_started",
        "message": f"Evaluation job submitted. Results will be saved to {request.output_dir or 'default location'}",
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# HEALTH & INFO
# ============================================

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": evaluator is not None,
        "current_model": current_model_path,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Get current model information"""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": str(evaluator.model_path),
        "num_labels": evaluator.num_labels,
        "device": str(evaluator.device),
        "label_encoder_classes": len(evaluator.le.classes_)
    }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
