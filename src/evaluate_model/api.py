# Evaluate API

import os
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Rakuten ML Evaluate API")

# Import your modules
from src.predict.predict_text import TextPredictor
from src.evaluate_model.evaluate_text import ModelEvaluator

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
