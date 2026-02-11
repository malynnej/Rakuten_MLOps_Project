# src/data/api.py
"""
Data Service API

"""

import os
import traceback
from datetime import datetime

import pandas as pd
from core.config import get_path, load_config
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from services.data_import.import_raw_data import import_raw_data
from services.preprocess.text_preparation_pipeline import TextPreparationPipeline

API_ROOT_PATH = "/data"

app = FastAPI(
    title="Data Service - Rakuten MLOps",
    root_path=API_ROOT_PATH,
)


# Workaround to make docs available behind proxy AND locally
@app.get(f"{API_ROOT_PATH}/openapi.json", include_in_schema=False)
async def get_docs():
    return RedirectResponse(url="/openapi.json")


# ============================================ GLOBAL STATE
# ============================================

# Initialize pipeline
pipeline = None

# Processing status tracker
processing_status = {
    "is_processing": False,
    "status": "idle",
    "last_processing": None,
    "results": None,
    "error_details": None,
    "progress": 0,
}


@app.on_event("startup")
async def startup():
    """Initialize pipeline at startup"""
    global pipeline
    try:
        pipeline = TextPreparationPipeline()
        print("✓ Data pipeline initialized at startup")
    except Exception as e:
        print(f"✗ Failed to initialize pipeline: {e}")
        pipeline = None


# ============================================ REQUEST MODELS
# ============================================


class PreprocessRequest(BaseModel):
    combine_existing_data: bool = False
    save_holdout: bool = True


class BatchPreprocessRequest(BaseModel):
    combine_existing_data: bool = False
    save_holdout: bool = True


# ============================================ RAW DATA IMPORT ENDPOINTS
# ============================================


@app.post("/import/raw")
async def import_raw_endpoint(background_tasks: BackgroundTasks):
    """
    Import raw data from S3 bucket. Downloads X_train and y_train CSV files.
    Runs in background to avoid timeout.

    Example:
        POST /import/raw
    """

    def import_job():
        try:
            print(f"\n{'=' * 60}")
            print("IMPORTING RAW DATA FROM S3")
            print(f"{'=' * 60}\n")

            # Get config
            paths = load_config("paths")
            raw_dir = get_path("data.raw")
            bucket_url = paths["data"]["bucket_raw"]
            filenames = [paths["data"]["X_train_raw"], paths["data"]["y_train_raw"]]

            print(f"Bucket: {bucket_url}")
            print(f"Files: {filenames}")
            print(f"Destination: {raw_dir}\n")

            # Import data
            import_raw_data(raw_dir, filenames, bucket_url)

            print("\n✓ Raw data imported successfully!")
            print(f"   Location: {raw_dir}")

        except Exception as e:
            print(f"✗ Import failed: {e}")
            traceback.print_exc()

    # Start background task
    background_tasks.add_task(import_job)

    return {
        "status": "import_started",
        "message": "Raw data import job submitted. Check logs for progress.",
        "timestamp": datetime.now().isoformat(),
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
                "size_mb": round(X_train_path.stat().st_size / 1024 / 1024, 2)
                if X_train_path.exists()
                else None,
            },
            "y_train": {
                "path": str(y_train_path),
                "exists": y_train_path.exists(),
                "size_mb": round(y_train_path.stat().st_size / 1024 / 1024, 2)
                if y_train_path.exists()
                else None,
            },
        },
        "timestamp": datetime.now().isoformat(),
    }


# ============================================ NEW CLASS DETECTION
# ============================================


@app.post("/preprocess/check-new-classes")
async def check_new_classes(file: UploadFile = File(...)):
    """
    Check if uploaded data contains new product classes.

    Use this before preprocessing to determine: - If label encoder must be
    re-fitted due to new classes

    Example:
        POST /preprocess/check-new-classes - Upload CSV file with 'prdtypecode'
        column

    Returns:
        Information about new classes and recommendations
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Read file
        df = pd.read_csv(file.file)

        if "prdtypecode" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'prdtypecode' column")

        # Check if label encoder exists
        models_dir = get_path("models.save_dir")
        encoder_path = models_dir / "label_encoder.pkl"

        if not encoder_path.exists():
            return {
                "has_existing_encoder": False,
                "has_new_classes": False,
                "message": "No existing label encoder found - this will be initial training",
                "recommendation": {"reason": "No existing model - initial training required"},
                "timestamp": datetime.now().isoformat(),
            }

        # USE THE PIPELINE'S METHOD
        has_new_classes, new_classes_info = pipeline._detect_new_classes(df)

        # Generate recommendation based on results
        if has_new_classes:
            recommendation = {
                "reason": """New classes detected - label encoder must be re-fitted and 
                model retrained from scratch"""
            }
        else:
            recommendation = {
                "reason": "No new classes - safe to fine-tune existing model with new data"
            }

        return {
            "has_existing_encoder": True,
            "has_new_classes": has_new_classes,
            "details": new_classes_info,
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": str(e), "traceback": traceback.format_exc()}
        )


# ============================================ PREPROCESSING ENDPOINTS
# ============================================


@app.post("/preprocess/from-raw")
async def preprocess_from_raw(request: PreprocessRequest, background_tasks: BackgroundTasks):
    """
    Preprocess data from raw CSV files in data/raw/.

    Automatically detects and handles new classes.

    Example:
        POST /preprocess/from-raw {
            "combine_existing_data": false, "save_holdout": true
        }
    """
    global processing_status, pipeline

    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="Pipeline not initialized. Check API startup logs."
        )

    if processing_status["is_processing"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Preprocessing already in progress",
                "current_status": processing_status["status"],
                "started_at": processing_status["last_processing"],
            },
        )

    # Validate prerequisites
    raw_dir = get_path("data.raw")
    paths = load_config("paths")
    X_train_path = raw_dir / paths["data"]["X_train_raw"]
    y_train_path = raw_dir / paths["data"]["y_train_raw"]

    if not (X_train_path.exists() and y_train_path.exists()):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Raw data files not found",
                "missing_files": {"X_train": str(X_train_path), "y_train": str(y_train_path)},
                "suggestion": "Run data import first: POST /import/raw",
            },
        )

    def preprocessing_job():
        global processing_status

        try:
            # Update status
            processing_status["is_processing"] = True
            processing_status["status"] = "processing"
            processing_status["last_processing"] = datetime.now().isoformat()
            processing_status["error_details"] = None
            processing_status["progress"] = 0

            print(f"\n{'=' * 60}")
            print("BACKGROUND PREPROCESSING STARTED")
            print(f"{'=' * 60}")
            print(f"Combine existing data: {request.combine_existing_data}")
            print(f"Save holdout: {request.save_holdout}")
            print(f"{'=' * 60}\n")

            # Load raw data
            X_train = pd.read_csv(X_train_path, index_col=0)
            y_train = pd.read_csv(y_train_path, index_col=0)

            # Combine
            df = X_train.join(y_train)
            print(f"Loaded {len(df):,} samples from raw data")

            processing_status["progress"] = 10

            # Preprocess with strategy
            output_paths = pipeline.prepare_training_data(
                df,
                combine_existing_data=request.combine_existing_data,
                save_holdout=request.save_holdout,
            )

            processing_status["progress"] = 90

            # Update status with success
            processing_status["is_processing"] = False
            processing_status["status"] = "completed"
            processing_status["progress"] = 100
            processing_status["results"] = output_paths

            print("\n✓ Background preprocessing completed!")
            print(f"   Train: {output_paths['num_train']:,} samples")
            print(f"   Val:   {output_paths['num_val']:,} samples")
            print(f"   Test:  {output_paths['num_test']:,} samples")

            if output_paths.get("has_new_classes"):
                print(" New classes detected and handled")

        except FileNotFoundError as e:
            processing_status["is_processing"] = False
            processing_status["status"] = "failed"
            processing_status["progress"] = 0
            processing_status["results"] = None
            processing_status["error_details"] = {
                "error_type": "FileNotFoundError",
                "message": str(e),
                "suggestion": "Run /import/raw first",
            }
            print(f"✗ Preprocessing failed: {e}")

        except Exception as e:
            processing_status["is_processing"] = False
            processing_status["status"] = "failed"
            processing_status["progress"] = 0
            processing_status["results"] = None
            processing_status["error_details"] = {
                "error_type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"✗ Preprocessing failed: {e}")
            traceback.print_exc()

    # Start background task
    background_tasks.add_task(preprocessing_job)

    return {
        "status": "preprocessing_started",
        "message": "Preprocessing job submitted. Monitor progress at /status",
        "config": {
            "combine_existing_data": request.combine_existing_data,
            "save_holdout": request.save_holdout,
        },
        "timestamp": datetime.now().isoformat(),
        "endpoints": {"status": "/status", "results": "/results/latest"},
    }


@app.post("/preprocess/batch")
async def preprocess_batch(
    file: UploadFile = File(...), combine_existing_data: bool = False, save_holdout: bool = False
):
    """
    Preprocess uploaded CSV file
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        # Validate file type
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read uploaded file
        df = pd.read_csv(file.file, index_col=0)

        print(f"\n{'=' * 60}")
        print(f"PROCESSING UPLOADED FILE: {file.filename}")
        print(f"{'=' * 60}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")
        print(f"Combine with existing data: {combine_existing_data}")
        print(f"Save holdout: {save_holdout}")
        print(f"{'=' * 60}\n")

        # Validate required columns
        required_cols = ["designation", "description", "prdtypecode"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

        # Preprocess with strategy
        output_paths = pipeline.prepare_training_data(
            df, combine_existing_data=combine_existing_data, save_holdout=save_holdout
        )

        return {
            "status": "success",
            "message": "Batch preprocessing complete",
            "input_file": file.filename,
            "config": {
                "combine_existing_data": combine_existing_data,
                "save_holdout": save_holdout,
            },
            "output_paths": output_paths,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail={"error": str(e), "traceback": traceback.format_exc()}
        )


# ============================================ STATUS & INFO ENDPOINTS
# ============================================


@app.get("/preprocessing/status")
async def get_status():
    """
    Get current preprocessing status with detailed progress.

    Returns:
        Current processing status, progress, and results if completed
    """
    return {"processing_status": processing_status, "timestamp": datetime.now().isoformat()}


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
        "timestamp": datetime.now().isoformat(),
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
            status_code=404, detail="No preprocessing results available. Run preprocessing first."
        )

    return {
        "status": "success",
        "results": processing_status["results"],
        "timestamp": processing_status["last_processing"],
    }


@app.get("/data_info")
async def data_info():
    """
    Get information about processed data files.

    Returns:
        File sizes and sample counts if data exists
    """
    preprocessed_dir = get_path("data.preprocessed")

    files = ["train.parquet", "val.parquet", "test.parquet", "holdout_raw.parquet"]
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
                "columns": list(df.columns),
            }
        else:
            file_info[filename] = {"path": str(filepath), "exists": False}

    return {"processed_data": file_info, "timestamp": datetime.now().isoformat()}


@app.get("/prerequisites")
async def check_prerequisites():
    """
    Check if all prerequisites for preprocessing are met.

    Returns:
        Status of required files and configuration
    """
    raw_dir = get_path("data.raw")
    preprocessed_dir = get_path("data.preprocessed")
    models_dir = get_path("models.save_dir")
    paths = load_config("paths")

    checks = {
        "raw_data": {
            "X_train": (raw_dir / paths["data"]["X_train_raw"]).exists(),
            "y_train": (raw_dir / paths["data"]["y_train_raw"]).exists(),
        },
        "preprocessed_data": {
            "train": (preprocessed_dir / "train.parquet").exists(),
            "val": (preprocessed_dir / "val.parquet").exists(),
            "test": (preprocessed_dir / "test.parquet").exists(),
            "holdout": (preprocessed_dir / "holdout_raw.parquet").exists(),
        },
        "label_encoder": {
            "encoder": (models_dir / "label_encoder.pkl").exists(),
            "mappings": (models_dir / "label_mappings.json").exists(),
        },
    }

    raw_ready = all(checks["raw_data"].values())
    preprocessed_ready = all(checks["preprocessed_data"].values())
    encoder_ready = all(checks["label_encoder"].values())

    return {
        "ready_for_initial_preprocessing": raw_ready,
        "ready_for_retraining": raw_ready and encoder_ready and preprocessed_ready,
        "has_preprocessed_data": preprocessed_ready,
        "has_label_encoder": encoder_ready,
        "checks": checks,
        "recommendations": {
            "initial_training": "POST /preprocess/from-raw with retrain=false"
            if raw_ready
            else "POST /import/raw first",
            "retraining": "POST /preprocess/from-raw with retrain=true"
            if (raw_ready and preprocessed_ready)
            else "Complete initial training first",
        },
        "timestamp": datetime.now().isoformat(),
    }


# ============================================ ROOT ENDPOINT
# ============================================


@app.get("/")
async def root():
    """API root with usage information"""
    return {
        "service": "Rakuten ML Data Service API",
        "version": "2.0.0",
        "description": "Data preprocessing",
        "features": [
            "Raw data import from S3",
            "Intelligent preprocessing pipeline",
            "New class detection",
            "Combine new with old data",
            "Holdout set management",
        ],
        "endpoints": "endpoints for accessing service",
        "docs": "/docs",
        "timestamp": datetime.now().isoformat(),
    }


# ============================================ RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
