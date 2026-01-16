# Train API

import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI(title="Rakuten ML Train API")

# Import your modules
from src.predict.predict_text import TextPredictor
from src.train_model.train_model_text import train_bert_model

# Global predictor (loaded once at startup)
predictor = None
models_dir = os.environ.get("MODELS_DIR", "./models")
model_name = os.environ.get("MODEL_NAME", "bert-rakuten-final")
current_model_path = os.path.join(models_dir, model_name)

@app.on_event("startup")
async def load_model():
    """Load model when API starts"""
    global predictor
    predictor = TextPredictor(current_model_path)
    print("Model loaded at startup")


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
