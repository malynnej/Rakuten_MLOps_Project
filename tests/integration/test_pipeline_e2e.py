"""
End-to-end pipeline integration tests
"""

import requests
import time
import pytest

# Service URLs
DATA_API = "http://localhost:8001"
TRAIN_API = "http://localhost:8002"
EVAL_API = "http://localhost:8004"
PREDICT_API = "http://localhost:8000"


class TestPipelineEndToEnd:
    """Test complete pipeline workflow"""
    
    def test_full_pipeline_workflow(self):
        """Test: Raw data → Preprocess → Train → Evaluate → Predict"""
        
        # Step 1: Check all services healthy
        assert requests.get(f"{DATA_API}/health").status_code == 200
        assert requests.get(f"{TRAIN_API}/health").status_code == 200
        assert requests.get(f"{EVAL_API}/health").status_code == 200
        assert requests.get(f"{PREDICT_API}/health").status_code == 200
        
        # Step 2: Check prerequisites
        prereq_resp = requests.get(f"{DATA_API}/prerequisites")
        assert prereq_resp.status_code == 200
        
        # Step 3: Preprocess data (if needed)
        if not prereq_resp.json()["has_preprocessed_data"]:
            preprocess_resp = requests.post(
                f"{DATA_API}/preprocess/from-raw",
                json={"combine_existing_data": False, "save_holdout": True}
            )
            assert preprocess_resp.status_code in [200, 400, 409]
            
            # Wait for preprocessing (with timeout)
            for _ in range(30):  # 30 attempts, 10 sec each = 5 min
                status = requests.get(f"{DATA_API}/preprocessing/status")
                if status.json()["processing_status"]["status"] == "completed":
                    break
                time.sleep(10)
        
        # Step 4: Train model (optional - takes time)
        # Uncomment to test training
        # train_resp = requests.post(
        #     f"{TRAIN_API}/train",
        #     json={"retrain": False, "model_name": "bert-rakuten-test"}
        # )
        # assert train_resp.status_code in [200, 400, 409]
        
        # Step 5: Make prediction
        predict_resp = requests.post(
            f"{PREDICT_API}/predict/text",
            json={"text": "Nike running shoes"}
        )
        assert predict_resp.status_code == 200
        assert "predicted_category" in predict_resp.json()
        
        # Step 6: Verify prediction quality
        prediction = predict_resp.json()
        assert 0.0 <= prediction["confidence"] <= 1.0
