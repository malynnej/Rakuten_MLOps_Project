# Complete API with all endpoints

from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

app = FastAPI(title="Rakuten ML Pipeline API")

# Import your modules
from src.models.predict_text import TextPredictor
from src.models.train_model_text import train_bert_model
from src.models.evaluate_text import ModelEvaluator

# Global predictor (loaded once at startup)
predictor = None
current_model_path = "./models/bert-rakuten-final"

@app.on_event("startup")
async def load_model():
    """Load model when API starts"""
    global predictor
    predictor = TextPredictor(current_model_path)
    print("Model loaded at startup")


# ============================================
# PREDICTION ENDPOINTS
# ============================================

class PredictRequest(BaseModel):
    text: str
    return_top5: bool = False

class BatchPredictRequest(BaseModel):
    texts: List[str]
    return_top5: bool = False

@app.post("/predict")
async def predict_single(request: PredictRequest):
    """Predict category for single text"""
    try:
        result = predictor.predict(request.text, return_probabilities=request.return_top5)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    """Predict categories for multiple texts"""
    try:
        results = predictor.predict(request.texts, return_probabilities=request.return_top5)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# EVALUATION ENDPOINT
# ============================================

class EvaluateRequest(BaseModel):
    dataset_path: str  # Path to tokenized test dataset
    output_dir: str = "./src/data/results/evaluation"
    batch_size: int = 16

@app.post("/evaluate")
async def evaluate_model(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """
    Evaluate model on test dataset and generate confusion matrix + classification report.
    Runs in background to avoid timeout.
    """
    def evaluation_job():
        try:
            from datasets import load_from_disk
            
            # Load dataset
            dataset = load_from_disk(request.dataset_path)
            test_dataset = dataset["test"]
            
            # Evaluate
            evaluator = ModelEvaluator(current_model_path)
            results = evaluator.evaluate_dataset(
                test_dataset,
                batch_size=request.batch_size,
                output_dir=request.output_dir
            )
            
            print(f"Evaluation complete - Results saved to {request.output_dir}")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
    
    background_tasks.add_task(evaluation_job)
    
    return {
        "status": "evaluation_started",
        "message": f"Evaluation job submitted. Results will be saved to {request.output_dir}",
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# RETRAINING ENDPOINT
# ============================================

retraining_status = {
    "is_running": False,
    "last_run": None,
    "last_result": None
}

class RetrainRequest(BaseModel):
    model_path: str = "./models/bert-rakuten-final"
    retrain_from_existing: bool = True
    version: Optional[str] = None

@app.post("/retrain")
async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    global retraining_status, predictor, current_model_path
    
    if retraining_status["is_running"]:
        raise HTTPException(status_code=409, detail="Retraining already in progress")
    
    def retrain_job():
        global retraining_status, predictor, current_model_path
        
        try:
            retraining_status["is_running"] = True
            retraining_status["started_at"] = datetime.now().isoformat()
            
            # Train model
            metrics = train_bert_model(
                retrain=request.retrain_from_existing,
                model_path=request.model_path,
                version=request.version
            )
            
            # Update status
            retraining_status["is_running"] = False
            retraining_status["last_run"] = datetime.now().isoformat()
            retraining_status["last_result"] = {
                "status": "success",
                "metrics": metrics
            }
            
            # Reload predictor with new model
            new_model_path = metrics.get("model_path", request.model_path)
            predictor = TextPredictor(new_model_path)
            current_model_path = new_model_path
            
            print(f"✅ Retraining complete - Model reloaded from {new_model_path}")
            
        except Exception as e:
            retraining_status["is_running"] = False
            retraining_status["last_run"] = datetime.now().isoformat()
            retraining_status["last_result"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"Retraining failed: {e}")
    
    background_tasks.add_task(retrain_job)
    
    return {
        "status": "retraining_started",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/retrain/status")
async def get_retrain_status():
    """Check retraining status"""
    return retraining_status


# ============================================
# DATA PIPELINE ENDPOINTS (Future)
# ============================================

@app.post("/data/import")
async def import_raw_data():
    """Trigger raw data import"""
    # Call import_raw_data.py function
    pass

@app.post("/data/preprocess")
async def preprocess_data():
    """Trigger preprocessing pipeline"""
    # Call preprocessing_pipeline.py function
    pass


# ============================================
# HEALTH & INFO
# ============================================

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "current_model": current_model_path,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info")
async def model_info():
    """Get current model information"""
    return {
        "model_path": current_model_path,
        "num_classes": len(predictor.le.classes_),
        "device": str(predictor.device)
    }
