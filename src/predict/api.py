# Predict API

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from datetime import datetime

app = FastAPI(title="Rakuten ML Predict API")

# Import your modules
from src.predict.predict_text import TextPredictor

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
