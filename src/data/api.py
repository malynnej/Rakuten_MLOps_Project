# src/data/api.py
"""
Data Service API

Handles:
1. Importing raw data from S3
2. Preprocessing data for training (with label encoding, splitting, tokenization)
3. Batch preprocessing from uploaded files
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from services.data_import.import_raw_data import import_raw_data
from services.preprocess.data_preparation import TrainingDataPipeline
from core.config import load_config, get_path

app = FastAPI(title="Rakuten ML Data Service API")


# ============================================
# GLOBAL STATE
# ============================================

# Initialize pipeline
pipeline = None

# Processing status tracker
processing_status = {
    "is_processing": False,
    "status": "idle",
    "last_processing": None,
    "results": None
}


@app.on_event("startup")
async def startup():
    """Initialize pipeline at startup"""
    global pipeline
    try:
        pipeline = TrainingDataPipeline()
        print(" Data pipeline initialized at startup")
    except Exception as e:
        print(f" Failed to initialize pipeline: {e}")
        pipeline = None


# ============================================
# REQUEST MODELS
# ============================================

class PreprocessRequest(BaseModel):
    retrain: bool = False
    save_holdout: bool = True


# ============================================
# RAW DATA IMPORT ENDPOINTS
# ============================================

@app.post("/import/raw")
async def import_raw_endpoint(background_tasks: BackgroundTasks):
    """
    Import raw data from S3 bucket.
    Downloads X_train and y_train CSV files.
    Runs in background to avoid timeout.
    
    Example:
        POST /import/raw
    """
    def import_job():
        try:
            print(f"\n{'='*60}")
            print("IMPORTING RAW DATA FROM S3")
            print(f"{'='*60}\n")
            
            # Import data
            import_raw_data()
            
            raw_dir = get_path("data.raw")
            print(f"\n Raw data imported successfully!")
            print(f"   Location: {raw_dir}")
            
        except Exception as e:
            print(f" Import failed: {e}")
    
    # Start background task
    background_tasks.add_task(import_job)
    
    return {
        "status": "import_started",
        "message": "Raw data import job submitted. Check logs for progress.",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/import/status")
async def import_status():
    """
    Check if raw data files exist.
    
    Returns:
        Status of raw data files
    """
    raw_dir = get_path("data.raw")
    paths = load_config("paths")
    
    X_train_path = raw_dir / paths["data"]["X_train_raw"]
    y_train_path = raw_dir / paths["data"]["y_train_raw"]
    
    return {
        "raw_data_exists": X_train_path.exists() and y_train_path.exists(),
        "files": {
            "X_train": {
                "path": str(X_train_path),
                "exists": X_train_path.exists(),
                "size_mb": round(X_train_path.stat().st_size / 1024 / 1024, 2) if X_train_path.exists() else None
            },
            "y_train": {
                "path": str(y_train_path),
                "exists": y_train_path.exists(),
                "size_mb": round(y_train_path.stat().st_size / 1024 / 1024, 2) if y_train_path.exists() else None
            }
        },
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# PREPROCESSING ENDPOINTS
# ============================================

@app.post("/preprocess/from-raw")
async def preprocess_from_raw(
    request: PreprocessRequest,
    background_tasks: BackgroundTasks
):
    """
    Preprocess data from raw CSV files in data/raw/.
    
    This is the main endpoint for training data preparation:
    - Loads raw X_train and y_train CSV files
    - Combines text columns
    - Cleans and preprocesses text
    - Encodes labels
    - Splits into train/val/test/holdout
    - Tokenizes
    - Saves as parquet files
    
    Example:
        POST /preprocess/from-raw
        {
            "retrain": false,
            "save_holdout": true
        }
    """
    global processing_status, pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Check API startup logs."
        )
    
    if processing_status["is_processing"]:
        raise HTTPException(
            status_code=409,
            detail="Preprocessing already in progress"
        )
    
    def preprocessing_job():
        global processing_status
        
        try:
            # Update status
            processing_status["is_processing"] = True
            processing_status["status"] = "processing"
            processing_status["last_processing"] = datetime.now().isoformat()
            
            print(f"\n{'='*60}")
            print("BACKGROUND PREPROCESSING STARTED")
            print(f"{'='*60}\n")
            
            # Load raw data
            raw_dir = get_path("data.raw")
            paths = load_config("paths")
            
            X_train = pd.read_csv(raw_dir / paths["data"]["X_train_raw"], index_col=0)
            y_train = pd.read_csv(raw_dir / paths["data"]["y_train_raw"], index_col=0)
            
            # Combine
            df = X_train.join(y_train)
            
            print(f"Loaded {len(df):,} samples from raw data")
            
            # Preprocess
            output_paths = pipeline.prepare_training_data(
                df,
                retrain=request.retrain,
                save_holdout=request.save_holdout
            )
            
            # Update status with success
            processing_status["is_processing"] = False
            processing_status["status"] = "completed"
            processing_status["results"] = output_paths
            
            print(f"\n Background preprocessing completed!")
            print(f"   Train: {output_paths['num_train']:,} samples")
            print(f"   Val:   {output_paths['num_val']:,} samples")
            print(f"   Test:  {output_paths['num_test']:,} samples")
            
        except FileNotFoundError as e:
            processing_status["is_processing"] = False
            processing_status["status"] = f"failed: Raw data files not found. Run /import/raw first."
            processing_status["results"] = None
            print(f" Preprocessing failed: {e}")
            
        except Exception as e:
            processing_status["is_processing"] = False
            processing_status["status"] = f"failed: {str(e)}"
            processing_status["results"] = None
            print(f" Preprocessing failed: {e}")
    
    # Start background task
    background_tasks.add_task(preprocessing_job)
    
    return {
        "status": "preprocessing_started",
        "message": "Preprocessing job submitted. Check /status for progress.",
        "retrain": request.retrain,
        "save_holdout": request.save_holdout,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/preprocess/batch")
async def preprocess_batch(
    file: UploadFile = File(...),
    retrain: bool = False,
    save_holdout: bool = True
):
    """
    Preprocess uploaded CSV file.
    
    Useful for custom datasets or testing.
    
    Example:
        POST /preprocess/batch
        - Upload CSV file with columns: designation, description, prdtypecode
        - Form data: retrain=false, save_holdout=true
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized"
        )
    
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported"
            )
        
        # Read uploaded file
        df = pd.read_csv(file.file, index_col=0)
        
        print(f"\n{'='*60}")
        print(f"PROCESSING UPLOADED FILE: {file.filename}")
        print(f"{'='*60}\n")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        
        # Validate required columns
        required_cols = ["designation", "description", "prdtypecode"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Preprocess
        output_paths = pipeline.prepare_training_data(
            df,
            retrain=retrain,
            save_holdout=save_holdout
        )
        
        return {
            "status": "success",
            "message": "Batch preprocessing complete",
            "input_file": file.filename,
            "output_paths": output_paths,
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# STATUS & INFO ENDPOINTS
# ============================================

@app.get("/status")
async def get_status():
    """
    Get current preprocessing status.
    
    Returns:
        Current processing status and results if completed
    """
    return {
        "processing_status": processing_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        API health status and pipeline state
    """
    return {
        "status": "healthy",
        "service": "data-service",
        "pipeline_initialized": pipeline is not None,
        "is_processing": processing_status["is_processing"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/results/latest")
async def get_latest_results():
    """
    Get results from most recent preprocessing.
    
    Returns:
        Latest preprocessing results or 404 if none available
    """
    if processing_status["results"] is None:
        raise HTTPException(
            status_code=404,
            detail="No preprocessing results available"
        )
    
    return {
        "status": "success",
        "results": processing_status["results"],
        "timestamp": processing_status["last_processing"]
    }


@app.get("/data/info")
async def data_info():
    """
    Get information about processed data files.
    
    Returns:
        File sizes and sample counts if data exists
    """
    preprocessed_dir = get_path("data.preprocessed")
    
    files = ["train.parquet", "val.parquet", "test.parquet", "holdout.parquet"]
    file_info = {}
    
    for filename in files:
        filepath = preprocessed_dir / filename
        if filepath.exists():
            df = pd.read_parquet(filepath)
            file_info[filename] = {
                "path": str(filepath),
                "exists": True,
                "size_mb": round(filepath.stat().st_size / 1024 / 1024, 2),
                "num_samples": len(df),
                "columns": list(df.columns)
            }
        else:
            file_info[filename] = {
                "path": str(filepath),
                "exists": False
            }
    
    return {
        "processed_data": file_info,
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# ROOT ENDPOINT
# ============================================

@app.get("/")
async def root():
    """API root with usage information"""
    return {
        "service": "Rakuten ML Data Service API",
        "version": "1.0.0",
        "endpoints": {
            "import": "POST /import/raw - Import raw data from S3",
            "import_status": "GET /import/status - Check raw data files",
            "preprocess_raw": "POST /preprocess/from-raw - Preprocess raw data",
            "preprocess_batch": "POST /preprocess/batch - Preprocess uploaded file",
            "status": "GET /status - Get preprocessing status",
            "health": "GET /health - Health check",
            "latest_results": "GET /results/latest - Latest preprocessing results",
            "data_info": "GET /data/info - Processed data information"
        },
        "docs": "/docs",
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
