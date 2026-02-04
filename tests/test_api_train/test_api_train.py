"""
Test suite for Training Service API

Tests all endpoints including:
- Model training workflows
- Status monitoring
- Prerequisites checking
- Health checks
- Results retrieval
"""

import logging
import os
import time
from datetime import datetime

import pytest
import requests

# API configuration
API_ADDRESS = os.environ.get("API_ADDRESS", default="127.0.0.1")
API_PORT = os.environ.get("API_PORT", default="8002")
BASE_URL = f"http://{API_ADDRESS}:{API_PORT}"

# Test configuration
# TEST_MODEL_NAME = "bert-rakuten-final"  # ← actual model
# Or for safety during testing:
 TEST_MODEL_NAME = "bert-rakuten-test"

# ============================================
# FIXTURES
# ============================================

@pytest.fixture(scope="session")
def api_base_url():
    """Fixture for API base URL"""
    return BASE_URL


# ============================================
# TEST ROOT ENDPOINT
# ============================================

class TestRoot:
    """Test class for root API endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on the root endpoint"""
        logging.info("Sending request to root endpoint")
        url = f"{api_base_url}/"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content from the root endpoint response"""
        logging.info("Extracting content from root endpoint response")
        return req.json()

    def test_root_status_code(self, req):
        """Test the status code of the root endpoint"""
        logging.info("Testing root endpoint status code")
        expected_status_code = 200
        assert req.status_code == expected_status_code, (
            f"Expected status code {expected_status_code}, got {req.status_code}"
        )

    @pytest.mark.parametrize(
        "key, expected_type",
        [
            ("service", str),
            ("version", str),
            ("description", str),
            ("endpoints", dict),
            ("docs", str),
            ("timestamp", str),
        ],
    )
    def test_keys_types(self, req_content, key, expected_type):
        """Test the returned content keys and types"""
        assert key in req_content, f"Key '{key}' not found in response"
        assert isinstance(req_content[key], expected_type), (
            f"Content type for '{key}' is not {expected_type}"
        )

    def test_timestamp_format(self, req_content):
        """Test the timestamp format"""
        try:
            datetime.fromisoformat(req_content["timestamp"])
        except ValueError:
            assert False, "Timestamp is not in valid ISO format"


# ============================================
# TEST HEALTH ENDPOINT
# ============================================

class TestHealth:
    """Test class for health API endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on the health endpoint"""
        logging.info("Sending request to health endpoint")
        url = f"{api_base_url}/health"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content from the health endpoint response"""
        logging.info("Extracting content from health endpoint response")
        return req.json()

    def test_health_status_code(self, req):
        """Test the status code of the health endpoint"""
        logging.info("Testing health endpoint status code")
        expected_status_code = 200
        assert req.status_code == expected_status_code, (
            f"Expected status code {expected_status_code}, got {req.status_code}"
        )

    @pytest.mark.parametrize(
        "key, expected_type",
        [
            ("status", str),
            ("service", str),
            ("is_training", bool),
            ("timestamp", str),
        ],
    )
    def test_health_keys_types(self, req_content, key, expected_type):
        """Test the type of the health endpoint response content"""
        assert key in req_content, f"Key '{key}' not found in response"
        assert isinstance(req_content[key], expected_type), (
            f"Content type for '{key}' is not {expected_type}"
        )

    def test_health_status_message(self, req_content):
        """Test the status message of the health endpoint response"""
        expected_status_str = "healthy"
        assert req_content["status"] == expected_status_str, (
            f"Expected status '{expected_status_str}', got '{req_content['status']}'"
        )

    def test_timestamp_format(self, req_content):
        """Test the timestamp format"""
        try:
            datetime.fromisoformat(req_content["timestamp"])
        except ValueError:
            assert False, "Timestamp is not in valid ISO format"


# ============================================
# TEST PREREQUISITES ENDPOINT
# ============================================

class TestPrerequisites:
    """Test prerequisites checking endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on prerequisites endpoint"""
        logging.info("Sending request to prerequisites endpoint")
        url = f"{api_base_url}/prerequisites"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content"""
        return req.json()

    def test_status_code(self, req):
        """Test status code"""
        assert req.status_code == 200

    @pytest.mark.parametrize(
        "key, expected_type",
        [
            ("ready_for_training", bool),
            ("checks", dict),
            ("timestamp", str),
        ],
    )
    def test_keys_types(self, req_content, key, expected_type):
        """Test response structure"""
        assert key in req_content, f"Key '{key}' not found"
        assert isinstance(req_content[key], expected_type), (
            f"Type for '{key}' is not {expected_type}"
        )

    def test_checks_structure(self, req_content):
        """Test checks structure"""
        checks = req_content["checks"]
        
        assert "data_preprocessed" in checks
        assert "label_encoder" in checks
        assert "models" in checks
        
        # Check data_preprocessed structure
        data_prep = checks["data_preprocessed"]
        assert "train" in data_prep
        assert "val" in data_prep
        assert "test" in data_prep
        
        # Check label_encoder structure
        label_enc = checks["label_encoder"]
        assert "encoder" in label_enc
        assert "mappings" in label_enc

    def test_timestamp_format(self, req_content):
        """Test timestamp format"""
        try:
            datetime.fromisoformat(req_content["timestamp"])
        except ValueError:
            assert False, "Timestamp is not in valid ISO format"


# ============================================
# TEST STATUS ENDPOINT
# ============================================

class TestStatus:
    """Test training status endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on status endpoint"""
        logging.info("Sending request to status endpoint")
        url = f"{api_base_url}/status"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content"""
        return req.json()

    def test_status_code(self, req):
        """Test status code"""
        assert req.status_code == 200

    def test_has_training_status(self, req_content):
        """Test that training_status exists"""
        assert "training_status" in req_content
        assert "timestamp" in req_content
        
        status = req_content["training_status"]
        assert "is_training" in status
        assert "status" in status
        assert "progress" in status
        
        assert isinstance(status["is_training"], bool)
        assert isinstance(status["status"], str)
        assert isinstance(status["progress"], (int, float))

    def test_status_valid_values(self, req_content):
        """Test that status has valid values"""
        status = req_content["training_status"]
        
        # Status should be one of these
        valid_statuses = ["idle", "training", "completed", "failed"]
        assert status["status"] in valid_statuses, (
            f"Status '{status['status']}' not in valid statuses"
        )
        
        # Progress should be 0-100
        progress = status["progress"]
        assert 0 <= progress <= 100, (
            f"Progress {progress} not in range [0, 100]"
        )


# ============================================
# TEST LATEST RESULTS ENDPOINT
# ============================================

class TestLatestResults:
    """Test latest results endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on latest results endpoint"""
        logging.info("Sending request to latest results endpoint")
        url = f"{api_base_url}/results/latest"
        response = requests.get(url=url)
        return response

    def test_status_code(self, req):
        """Test status code is either 200 or 404"""
        # 200 if results exist, 404 if no training yet
        assert req.status_code in [200, 404], (
            f"Unexpected status code {req.status_code}"
        )

    def test_response_structure_when_available(self, req):
        """Test response structure when results are available"""
        if req.status_code == 200:
            content = req.json()
            assert "status" in content
            assert "metrics" in content
            assert "training_completed" in content
            assert "timestamp" in content
            
            # Check metrics structure
            metrics = content["metrics"]
            assert isinstance(metrics, dict)

    def test_error_message_when_unavailable(self, req):
        """Test error message when results not available"""
        if req.status_code == 404:
            content = req.json()
            assert "detail" in content
            assert "training" in content["detail"].lower()


# ============================================
# TEST TRAIN ENDPOINT
# ============================================

class TestTrain:
    """Test training endpoint"""

    url_suffix = "/train"

    def test_train_request_structure(self, api_base_url):
        """Test training request accepts proper structure"""
        logging.info("Testing train endpoint request structure")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "retrain": False,
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should return 200 (started), 400 (missing data), or 409 (already training)
        assert response.status_code in [200, 400, 409], (
            f"Unexpected status code {response.status_code}"
        )

    def test_missing_training_data(self, api_base_url):
        """Test training when data is missing"""
        logging.info("Testing train endpoint with missing data")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "retrain": False,
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        if response.status_code == 400:
            content = response.json()
            assert "detail" in content
            detail = content["detail"]
            
            # Should mention missing data or files
            if isinstance(detail, dict):
                assert "error" in detail
            elif isinstance(detail, str):
                assert "data" in detail.lower() or "file" in detail.lower()

    def test_response_structure_when_started(self, api_base_url):
        """Test response structure when training starts"""
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "retrain": False,
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        if response.status_code == 200:
            content = response.json()
            assert "status" in content
            assert "message" in content
            assert "retrain" in content
            assert "model_name" in content
            assert "timestamp" in content
            assert "endpoints" in content
            
            # Verify returned values match request
            assert content["retrain"] == payload["retrain"]
            assert content["model_name"] == payload["model_name"]

    def test_retrain_without_model(self, api_base_url):
        """Test retraining when model doesn't exist"""
        logging.info("Testing retrain without existing model")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "retrain": True,
            "model_name": "nonexistent-model-xyz"
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should return 400 (model not found) or 409 (already training)
        if response.status_code == 400:
            content = response.json()
            assert "detail" in content
            detail = content["detail"]
            
            if isinstance(detail, dict):
                assert "error" in detail
                assert "model" in detail["error"].lower()

    def test_concurrent_training_prevention(self, api_base_url):
        """Test that concurrent training is prevented"""
        logging.info("Testing concurrent training prevention")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "retrain": False,
            "model_name": "bert-rakuten-concurrent-test"
        }
        
        # Send first request
        response1 = requests.post(url=url, json=payload)
        
        # Immediately send second request
        response2 = requests.post(url=url, json=payload)
        
        # At least one should work or properly reject
        statuses = {response1.status_code, response2.status_code}
        
        # Valid status codes
        assert all(s in [200, 400, 409] for s in statuses)
        
        # If one succeeded and training started, the other should be 409
        if 200 in statuses:
            # Check if the other is conflict
            other_status = response2.status_code if response1.status_code == 200 else response1.status_code
            # Other might be 409 or also 200 if first finished quickly
            assert other_status in [200, 409]

    def test_default_values(self, api_base_url):
        """Test training with default values"""
        logging.info("Testing train endpoint with defaults")
        url = f"{api_base_url}{self.url_suffix}"
        
        # Empty payload should use defaults
        payload = {}
        
        response = requests.post(url=url, json=payload)
        
        # Should accept (200/400/409) not validation error (422)
        assert response.status_code != 422, (
            "Should accept empty payload with defaults"
        )

    def test_invalid_model_name(self, api_base_url):
        """Test training with invalid model name"""
        logging.info("Testing train endpoint with invalid model name")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "retrain": False,
            "model_name": ""  # Empty model name
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should either reject or handle gracefully
        assert response.status_code in [200, 400, 409, 422]


# ============================================
# TEST EDGE CASES & ERROR HANDLING
# ============================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_retrain_type(self, api_base_url):
        """Test training with invalid retrain value"""
        logging.info("Testing invalid retrain type")
        url = f"{api_base_url}/train"
        
        payload = {
            "retrain": "yes",  # Should be boolean
            "model_name": "test"
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_extra_fields_ignored(self, api_base_url):
        """Test that extra fields are ignored"""
        logging.info("Testing extra fields in request")
        url = f"{api_base_url}/train"
        
        payload = {
            "retrain": False,
            "model_name": "test",
            "extra_field": "should be ignored"
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should work (extra fields ignored by Pydantic)
        assert response.status_code in [200, 400, 409]

    def test_malformed_json(self, api_base_url):
        """Test with malformed JSON"""
        logging.info("Testing malformed JSON")
        url = f"{api_base_url}/train"
        
        # Send malformed JSON
        headers = {"Content-Type": "application/json"}
        malformed_json = '{"retrain": false, "model_name": "test"'  # Missing closing brace
        
        response = requests.post(url=url, data=malformed_json, headers=headers)
        
        # Should return 422 (validation error)
        assert response.status_code == 422


# ============================================
# TEST INTEGRATION WORKFLOW
# ============================================

class TestAPIWorkflow:
    """Test complete API workflow"""

    def test_complete_workflow(self, api_base_url):
        """Test complete training API workflow"""
        logging.info("Testing complete API workflow")
        
        # Step 1: Check health
        health_url = f"{api_base_url}/health"
        health_resp = requests.get(url=health_url)
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"
        logging.info("✓ Health check passed")
        
        # Step 2: Check prerequisites
        prereq_url = f"{api_base_url}/prerequisites"
        prereq_resp = requests.get(url=prereq_url)
        assert prereq_resp.status_code == 200
        logging.info("✓ Prerequisites check passed")
        
        # Step 3: Check training status
        status_url = f"{api_base_url}/status"
        status_resp = requests.get(url=status_url)
        assert status_resp.status_code == 200
        logging.info("✓ Status check passed")
        
        # Step 4: Check latest results (may be 404 if no training yet)
        results_url = f"{api_base_url}/results/latest"
        results_resp = requests.get(url=results_url)
        assert results_resp.status_code in [200, 404]
        logging.info("✓ Results check passed")
        
        logging.info("✓ Complete workflow test passed")

    def test_status_consistency(self, api_base_url):
        """Test that status endpoint is consistent"""
        logging.info("Testing status consistency")
        
        status_url = f"{api_base_url}/status"
        
        # Get status twice
        resp1 = requests.get(url=status_url)
        time.sleep(0.1)  # Small delay
        resp2 = requests.get(url=status_url)
        
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        
        status1 = resp1.json()["training_status"]
        status2 = resp2.json()["training_status"]
        
        # If not training, status should be same
        if not status1["is_training"]:
            assert status1["status"] == status2["status"]


# ============================================
# TEST METRICS STRUCTURE (IF TRAINING COMPLETED)
# ============================================

class TestMetricsStructure:
    """Test metrics structure when available"""

    def test_metrics_fields(self, api_base_url):
        """Test metrics contain expected fields"""
        url = f"{api_base_url}/results/latest"
        response = requests.get(url=url)
        
        if response.status_code == 200:
            content = response.json()
            metrics = content["metrics"]
            
            # Expected metric fields
            expected_fields = [
                "test_accuracy",
                "num_labels",
                "device",
                "mode",
                "model_path",
            ]
            
            for field in expected_fields:
                assert field in metrics, f"Field '{field}' missing from metrics"

    def test_metrics_value_types(self, api_base_url):
        """Test metrics have correct value types"""
        url = f"{api_base_url}/results/latest"
        response = requests.get(url=url)
        
        if response.status_code == 200:
            metrics = response.json()["metrics"]
            
            # Test accuracy is float between 0 and 1
            if metrics.get("test_accuracy") is not None:
                accuracy = metrics["test_accuracy"]
                assert isinstance(accuracy, (int, float))
                assert 0.0 <= accuracy <= 1.0, (
                    f"Accuracy {accuracy} not in range [0, 1]"
                )
            
            # Num labels is positive integer
            if metrics.get("num_labels") is not None:
                num_labels = metrics["num_labels"]
                assert isinstance(num_labels, int)
                assert num_labels > 0

            # Device is string
            if metrics.get("device") is not None:
                device = metrics["device"]
                assert isinstance(device, str)
                assert device in ["cpu", "cuda", "mps", "cuda:0"]


# ============================================
# PYTEST CONFIGURATION
# ============================================

@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Setup logging for tests"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

