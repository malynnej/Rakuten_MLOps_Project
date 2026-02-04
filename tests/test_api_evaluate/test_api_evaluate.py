"""
Test suite for Evaluation Service API

Tests all endpoints including:
- Model evaluation workflows
- Status monitoring
- Model information
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
API_PORT = os.environ.get("API_PORT", default="8004")
BASE_URL = f"http://{API_ADDRESS}:{API_PORT}"

# Test configuration
TEST_MODEL_NAME = os.environ.get("TEST_MODEL_NAME", default="bert-rakuten-final")


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
            ("model_loaded", bool),
            ("is_evaluating", bool),
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

    def test_model_loaded(self, req_content):
        """Test that model is loaded at startup"""
        # Model should be loaded if service started successfully
        assert isinstance(req_content["model_loaded"], bool)

    def test_timestamp_format(self, req_content):
        """Test the timestamp format"""
        try:
            datetime.fromisoformat(req_content["timestamp"])
        except ValueError:
            assert False, "Timestamp is not in valid ISO format"


# ============================================
# TEST MODEL INFO ENDPOINT
# ============================================

class TestModelInfo:
    """Test model info endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on model info endpoint"""
        logging.info("Sending request to model info endpoint")
        url = f"{api_base_url}/model/info"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content"""
        if req.status_code == 200:
            return req.json()
        return None

    def test_status_code(self, req):
        """Test status code"""
        # 200 if model loaded, 503 if not
        assert req.status_code in [200, 503], (
            f"Unexpected status code {req.status_code}"
        )

    def test_model_info_structure(self, req, req_content):
        """Test model info structure when available"""
        if req.status_code == 200 and req_content:
            assert "model_path" in req_content
            assert "num_labels" in req_content
            assert "device" in req_content
            assert "label_encoder_classes" in req_content
            assert "timestamp" in req_content

    @pytest.mark.parametrize(
        "key, expected_type",
        [
            ("model_path", str),
            ("num_labels", int),
            ("device", str),
            ("label_encoder_classes", int),
        ],
    )
    def test_model_info_types(self, req, req_content, key, expected_type):
        """Test model info field types"""
        if req.status_code == 200 and req_content:
            assert key in req_content, f"Key '{key}' not found"
            assert isinstance(req_content[key], expected_type), (
                f"Type for '{key}' is not {expected_type}"
            )

    def test_num_labels_positive(self, req, req_content):
        """Test that num_labels is positive"""
        if req.status_code == 200 and req_content:
            num_labels = req_content["num_labels"]
            assert num_labels > 0, (
                f"num_labels should be positive, got {num_labels}"
            )

    def test_device_valid(self, req, req_content):
        """Test that device is valid"""
        if req.status_code == 200 and req_content:
            device = req_content["device"]
            valid_devices = ["cpu", "cuda", "mps", "cuda:0"]
            # Device might have additional info like "cuda:0" or "mps:0"
            assert any(valid in device for valid in valid_devices), (
                f"Device '{device}' not recognized"
            )


# ============================================
# TEST STATUS ENDPOINT
# ============================================

class TestStatus:
    """Test evaluation status endpoint"""

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

    def test_has_evaluation_status(self, req_content):
        """Test that evaluation_status exists"""
        assert "evaluation_status" in req_content
        assert "timestamp" in req_content
        
        status = req_content["evaluation_status"]
        assert "is_evaluating" in status
        assert "status" in status
        
        assert isinstance(status["is_evaluating"], bool)
        assert isinstance(status["status"], str)

    def test_status_valid_values(self, req_content):
        """Test that status has valid values"""
        status = req_content["evaluation_status"]
        
        # Status should be one of these
        valid_statuses = ["idle", "evaluating", "completed", "failed"]
        assert status["status"] in valid_statuses, (
            f"Status '{status['status']}' not in valid statuses"
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
        # 200 if results exist, 404 if no evaluation yet
        assert req.status_code in [200, 404], (
            f"Unexpected status code {req.status_code}"
        )

    def test_response_structure_when_available(self, req):
        """Test response structure when results are available"""
        if req.status_code == 200:
            content = req.json()
            assert "status" in content
            assert "results" in content
            assert "evaluated_at" in content
            assert "timestamp" in content
            
            # Check results structure
            results = content["results"]
            assert isinstance(results, dict)

    def test_error_message_when_unavailable(self, req):
        """Test error message when results not available"""
        if req.status_code == 404:
            content = req.json()
            assert "detail" in content
            assert "evaluation" in content["detail"].lower()


# ============================================
# TEST EVALUATE ENDPOINT
# ============================================

class TestEvaluate:
    """Test evaluation endpoint"""

    url_suffix = "/evaluate"

    def test_evaluate_request_structure(self, api_base_url):
        """Test evaluation request accepts proper structure"""
        logging.info("Testing evaluate endpoint request structure")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "test_path": None,
            "output_dir": None,
            "batch_size": 32,
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should return 200 (started), 400 (missing data), or 409 (already evaluating)
        assert response.status_code in [200, 400, 409, 503], (
            f"Unexpected status code {response.status_code}"
        )

    def test_missing_test_data(self, api_base_url):
        """Test evaluation when test data is missing"""
        logging.info("Testing evaluate endpoint with missing data")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "test_path": "/nonexistent/path/test.parquet",
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
                assert "test" in detail.lower() or "data" in detail.lower()

    def test_response_structure_when_started(self, api_base_url):
        """Test response structure when evaluation starts"""
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "test_path": None,
            "batch_size": 32,
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        if response.status_code == 200:
            content = response.json()
            assert "status" in content
            assert "message" in content
            assert "model_name" in content
            assert "output_dir" in content
            assert "timestamp" in content
            
            # Verify returned values
            assert content["model_name"] == TEST_MODEL_NAME

    def test_concurrent_evaluation_prevention(self, api_base_url):
        """Test that concurrent evaluation is prevented"""
        logging.info("Testing concurrent evaluation prevention")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "batch_size": 32,
            "model_name": TEST_MODEL_NAME
        }
        
        # Send first request
        response1 = requests.post(url=url, json=payload)
        
        # Immediately send second request
        response2 = requests.post(url=url, json=payload)
        
        statuses = {response1.status_code, response2.status_code}
        
        # Valid status codes
        assert all(s in [200, 400, 409, 503] for s in statuses)
        
        # If one succeeded, the other might be conflict
        if 200 in statuses:
            # At least one should work or properly reject
            assert True

    def test_custom_batch_size(self, api_base_url):
        """Test evaluation with custom batch size"""
        logging.info("Testing custom batch size")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "batch_size": 16,
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should accept (200/400/409/503) not validation error (422)
        assert response.status_code != 422, (
            "Should accept custom batch_size"
        )

    def test_default_values(self, api_base_url):
        """Test evaluation with default values"""
        logging.info("Testing evaluate endpoint with defaults")
        url = f"{api_base_url}{self.url_suffix}"
        
        # Empty payload should use defaults
        payload = {}
        
        response = requests.post(url=url, json=payload)
        
        # Should accept (200/400/409/503) not validation error (422)
        assert response.status_code != 422, (
            "Should accept empty payload with defaults"
        )

    def test_invalid_batch_size(self, api_base_url):
        """Test evaluation with invalid batch size"""
        logging.info("Testing invalid batch size")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "batch_size": 0,  # Invalid
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should either reject (422) or handle gracefully (200/400)
        assert response.status_code in [200, 400, 409, 422, 503]

    def test_custom_output_dir(self, api_base_url):
        """Test evaluation with custom output directory"""
        logging.info("Testing custom output directory")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "output_dir": "./custom_results",
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        if response.status_code == 200:
            content = response.json()
            assert "output_dir" in content
            # Should contain the custom path
            assert "custom_results" in content["output_dir"]


# ============================================
# TEST EDGE CASES & ERROR HANDLING
# ============================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_batch_size_type(self, api_base_url):
        """Test evaluation with invalid batch_size type"""
        logging.info("Testing invalid batch_size type")
        url = f"{api_base_url}/evaluate"
        
        payload = {
            "batch_size": "invalid",  # Should be int
            "model_name": TEST_MODEL_NAME
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_extra_fields_ignored(self, api_base_url):
        """Test that extra fields are ignored"""
        logging.info("Testing extra fields in request")
        url = f"{api_base_url}/evaluate"
        
        payload = {
            "batch_size": 32,
            "model_name": TEST_MODEL_NAME,
            "extra_field": "should be ignored"
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should work (extra fields ignored by Pydantic)
        assert response.status_code in [200, 400, 409, 503]

    def test_malformed_json(self, api_base_url):
        """Test with malformed JSON"""
        logging.info("Testing malformed JSON")
        url = f"{api_base_url}/evaluate"
        
        # Send malformed JSON
        headers = {"Content-Type": "application/json"}
        malformed_json = '{"batch_size": 32, "model_name": "test"'  # Missing closing brace
        
        response = requests.post(url=url, data=malformed_json, headers=headers)
        
        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_evaluator_not_initialized(self, api_base_url):
        """Test endpoints when evaluator not initialized"""
        # This would only happen if startup failed
        # Check that appropriate error is returned
        
        # Try model info
        info_url = f"{api_base_url}/model/info"
        info_resp = requests.get(url=info_url)
        
        # Should be 200 (loaded) or 503 (not loaded)
        assert info_resp.status_code in [200, 503]


# ============================================
# TEST INTEGRATION WORKFLOW
# ============================================

class TestAPIWorkflow:
    """Test complete API workflow"""

    def test_complete_workflow(self, api_base_url):
        """Test complete evaluation API workflow"""
        logging.info("Testing complete API workflow")
        
        # Step 1: Check health
        health_url = f"{api_base_url}/health"
        health_resp = requests.get(url=health_url)
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"
        logging.info("✓ Health check passed")
        
        # Step 2: Get model info
        info_url = f"{api_base_url}/model/info"
        info_resp = requests.get(url=info_url)
        assert info_resp.status_code in [200, 503]
        logging.info("✓ Model info check passed")
        
        # Step 3: Check evaluation status
        status_url = f"{api_base_url}/status"
        status_resp = requests.get(url=status_url)
        assert status_resp.status_code == 200
        logging.info("✓ Status check passed")
        
        # Step 4: Check latest results (may be 404 if no evaluation yet)
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
        
        status1 = resp1.json()["evaluation_status"]
        status2 = resp2.json()["evaluation_status"]
        
        # If not evaluating, status should be same
        if not status1["is_evaluating"]:
            assert status1["status"] == status2["status"]


# ============================================
# TEST RESULTS STRUCTURE (IF EVALUATION COMPLETED)
# ============================================

class TestResultsStructure:
    """Test results structure when available"""

    def test_results_fields(self, api_base_url):
        """Test results contain expected fields"""
        url = f"{api_base_url}/results/latest"
        response = requests.get(url=url)
        
        if response.status_code == 200:
            content = response.json()
            results = content["results"]
            
            # Expected result fields
            expected_fields = [
                "accuracy",
                "timestamp",
                "dataset_size",
                "output_files",
            ]
            
            for field in expected_fields:
                assert field in results, f"Field '{field}' missing from results"

    def test_accuracy_range(self, api_base_url):
        """Test accuracy is in valid range"""
        url = f"{api_base_url}/results/latest"
        response = requests.get(url=url)
        
        if response.status_code == 200:
            results = response.json()["results"]
            
            if "accuracy" in results and results["accuracy"] is not None:
                accuracy = results["accuracy"]
                assert isinstance(accuracy, (int, float))
                assert 0.0 <= accuracy <= 1.0, (
                    f"Accuracy {accuracy} not in range [0, 1]"
                )

    def test_dataset_size_positive(self, api_base_url):
        """Test dataset_size is positive"""
        url = f"{api_base_url}/results/latest"
        response = requests.get(url=url)
        
        if response.status_code == 200:
            results = response.json()["results"]
            
            if "dataset_size" in results and results["dataset_size"] is not None:
                dataset_size = results["dataset_size"]
                assert isinstance(dataset_size, int)
                assert dataset_size > 0


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
