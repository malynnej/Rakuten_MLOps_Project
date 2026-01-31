"""
Prediction API - Real-time inference service

Handles:
- Single text prediction
- Product prediction (designation + description)
- Batch prediction
"""
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.predict_text import PredictionService

app = FastAPI(title="Prediction Service - Rakuten MLOps")

# Initialize prediction service at startup
predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize predictor when API starts"""
    global predictor
    print("Initializing prediction service...")
    predictor = PredictionService()
    print("Prediction service ready!")


class TextInput(BaseModel):
    text: str
    return_probabilities: bool = False
    top_k: int = 5


class ProductInput(BaseModel):
    designation: str
    description: Optional[str] = ""
    return_probabilities: bool = False
    top_k: int = 5


class BatchTextInput(BaseModel):
    texts: List[str]
    batch_size: int = 32
    return_probabilities: bool = False
    top_k: int = 5

# Root endpoint
@app.get("/")
async def root():
    """API root with usage information"""
    return {
        "service": "Rakuten ML Prediction API",
        "version": "1.0.0",
        "description": "Real-time product category prediction",
        "endpoints": {
            "predict": "POST /predict - Predict from raw text",
            "predict_product": "POST /predict/product - Predict from designation + description",
            "predict_batch": "POST /predict/batch - Batch prediction",
            "model_info": "GET /model/info - Model information",
            "health": "GET /health - Health check"
        },
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "prediction-service",
        "model_loaded": predictor is not None,
        "device": str(predictor.device) if predictor else None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict/text")
async def predict_text(input_data: TextInput):
    """
    Predict category for single text.
    
    Example:
        POST /predict/text
        {
            "text": "Nike running shoes",
            "return_probabilities": true,
            "top_k": 3
        }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        result = predictor.predict(
            input_data.text,
            return_probabilities=input_data.return_probabilities,
            top_k=input_data.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/product")
async def predict_product(input_data: ProductInput):
    """
    Predict category for product with designation and description.
    
    Example:
        POST /predict/product
        {
            "designation": "Nike Air Max 90",
            "description": "Classic running shoes",
            "return_probabilities": true
        }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        result = predictor.predict_product(
            input_data.designation,
            input_data.description,
            return_probabilities=input_data.return_probabilities,
            top_k=input_data.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(input_data: BatchTextInput):
    """
    Predict categories for multiple texts.
    
    Example:
        POST /predict/batch
        {
            "texts": ["Nike shoes", "Adidas jacket", "Samsung phone"],
            "batch_size": 32,
            "return_probabilities": false
        }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        results = predictor.predict_batch(
            input_data.texts,
            batch_size=input_data.batch_size,
            return_probabilities=input_data.return_probabilities,
            top_k=input_data.top_k
        )
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

