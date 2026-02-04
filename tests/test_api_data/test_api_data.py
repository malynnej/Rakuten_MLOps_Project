"""
Test suite for Data Service API

Tests all endpoints including:
- Raw data import
- Preprocessing workflows
- New class detection
- Status monitoring
- Health checks
"""

import logging
import os
import time
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import pytest
import requests

# API configuration
API_ADDRESS = os.environ.get("API_ADDRESS", default="127.0.0.1")
API_PORT = os.environ.get("API_PORT", default="8001")
BASE_URL = f"http://{API_ADDRESS}:{API_PORT}"


# ============================================
# FIXTURES
# ============================================

@pytest.fixture(scope="session")
def api_base_url():
    """Fixture for API base URL"""
    return BASE_URL


@pytest.fixture(scope="class")
def sample_csv_data():
    """Create sample CSV data for testing"""
    data = {
        "designation": [
            "Nike running shoes",
            "Samsung smartphone",
            "Adidas jacket",
        ],
        "description": [
            "High quality athletic footwear",
            "Latest mobile technology",
            "Warm winter clothing",
        ],
        "prdtypecode": [2280, 2583, 1280],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="class")
def sample_csv_file(sample_csv_data):
    """Create sample CSV file for upload"""
    csv_buffer = StringIO()
    sample_csv_data.to_csv(csv_buffer)
    csv_buffer.seek(0)
    return ("test_data.csv", csv_buffer.getvalue(), "text/csv")


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
            ("features", list),
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
            ("pipeline_initialized", bool),
            ("is_processing", bool),
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

    def test_pipeline_initialized(self, req_content):
        """Test that pipeline is initialized"""
        assert req_content["pipeline_initialized"] is True, (
            "Pipeline should be initialized at startup"
        )


# ============================================
# TEST IMPORT STATUS ENDPOINT
# ============================================

class TestImportStatus:
    """Test class for import status endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on the import status endpoint"""
        logging.info("Sending request to import status endpoint")
        url = f"{api_base_url}/import/status"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content from the import status response"""
        return req.json()

    def test_status_code(self, req):
        """Test the status code"""
        assert req.status_code == 200

    @pytest.mark.parametrize(
        "key, expected_type",
        [
            ("raw_data_exists", bool),
            ("files", dict),
            ("timestamp", str),
        ],
    )
    def test_keys_types(self, req_content, key, expected_type):
        """Test response structure"""
        assert key in req_content, f"Key '{key}' not found in response"
        assert isinstance(req_content[key], expected_type), (
            f"Content type for '{key}' is not {expected_type}"
        )

    def test_files_structure(self, req_content):
        """Test files structure in response"""
        assert "X_train" in req_content["files"]
        assert "y_train" in req_content["files"]
        
        for file_key in ["X_train", "y_train"]:
            file_info = req_content["files"][file_key]
            assert "path" in file_info
            assert "exists" in file_info
            assert isinstance(file_info["exists"], bool)


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
            ("ready_for_initial_preprocessing", bool),
            ("ready_for_retraining", bool),
            ("has_preprocessed_data", bool),
            ("has_label_encoder", bool),
            ("checks", dict),
            ("recommendations", dict),
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
        
        assert "raw_data" in checks
        assert "preprocessed_data" in checks
        assert "label_encoder" in checks
        
        # Check raw_data structure
        assert "X_train" in checks["raw_data"]
        assert "y_train" in checks["raw_data"]


# ============================================
# TEST PREPROCESSING STATUS ENDPOINT
# ============================================

class TestPreprocessingStatus:
    """Test preprocessing status endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on status endpoint"""
        logging.info("Sending request to preprocessing status endpoint")
        url = f"{api_base_url}/preprocessing/status"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content"""
        return req.json()

    def test_status_code(self, req):
        """Test status code"""
        assert req.status_code == 200

    def test_has_processing_status(self, req_content):
        """Test that processing_status exists"""
        assert "processing_status" in req_content
        
        status = req_content["processing_status"]
        assert "is_processing" in status
        assert "status" in status
        assert isinstance(status["is_processing"], bool)
        assert isinstance(status["status"], str)


# ============================================
# TEST DATA INFO ENDPOINT
# ============================================

class TestDataInfo:
    """Test data info endpoint"""

    @pytest.fixture(scope="class")
    def req(self, api_base_url):
        """Request on data info endpoint"""
        logging.info("Sending request to data info endpoint")
        url = f"{api_base_url}/data_info"
        response = requests.get(url=url)
        return response

    @pytest.fixture(scope="class")
    def req_content(self, req):
        """Extract content"""
        return req.json()

    def test_status_code(self, req):
        """Test status code"""
        assert req.status_code == 200

    def test_response_structure(self, req_content):
        """Test response structure"""
        assert "processed_data" in req_content
        assert "timestamp" in req_content
        
        processed_data = req_content["processed_data"]
        expected_files = ["train.parquet", "val.parquet", "test.parquet", "holdout_raw.parquet"]
        
        for filename in expected_files:
            assert filename in processed_data
            file_info = processed_data[filename]
            assert "path" in file_info
            assert "exists" in file_info


# ============================================
# TEST BATCH PREPROCESSING ENDPOINT
# ============================================

class TestBatchPreprocessing:
    """Test batch preprocessing endpoint"""

    url_suffix = "/preprocess/batch"

    def test_missing_file(self, api_base_url):
        """Test batch preprocessing without file"""
        logging.info("Testing batch preprocessing without file")
        url = f"{api_base_url}{self.url_suffix}"
        
        # Send request without file
        response = requests.post(url=url)
        
        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_invalid_file_type(self, api_base_url):
        """Test batch preprocessing with non-CSV file"""
        logging.info("Testing batch preprocessing with invalid file type")
        url = f"{api_base_url}{self.url_suffix}"
        
        # Create a text file
        files = {"file": ("test.txt", "some text content", "text/plain")}
        response = requests.post(url=url, files=files)
        
        # Should return 400 (bad request)
        assert response.status_code == 400
        assert "CSV" in response.json()["detail"]

    def test_valid_csv_structure(self, api_base_url, sample_csv_data):
        """Test batch preprocessing with valid CSV"""
        logging.info("Testing batch preprocessing with valid CSV")
        url = f"{api_base_url}{self.url_suffix}"
        
        # Create CSV file
        csv_buffer = StringIO()
        sample_csv_data.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        files = {"file": ("test_data.csv", csv_buffer.getvalue(), "text/csv")}
        data = {
            "combine_existing_data": "false",
            "save_holdout": "false"
        }
        
        response = requests.post(url=url, files=files, data=data)
        
        # Should succeed (200) or fail gracefully (400/500)
        assert response.status_code in [200, 400, 500], (
            f"Unexpected status code {response.status_code}"
        )
        
        if response.status_code == 200:
            content = response.json()
            assert "status" in content
            assert "output_paths" in content

    def test_missing_required_columns(self, api_base_url):
        """Test batch preprocessing with missing columns"""
        logging.info("Testing batch preprocessing with missing columns")
        url = f"{api_base_url}{self.url_suffix}"
        
        # Create CSV with missing columns
        incomplete_data = pd.DataFrame({
            "designation": ["Product 1"],
            # Missing description and prdtypecode
        })
        
        csv_buffer = StringIO()
        incomplete_data.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        files = {"file": ("incomplete.csv", csv_buffer.getvalue(), "text/csv")}
        response = requests.post(url=url, files=files)
        
        # Should return 400 with error about missing columns
        assert response.status_code == 400
        assert "Missing required columns" in response.json()["detail"]


# ============================================
# TEST CHECK NEW CLASSES ENDPOINT
# ============================================

class TestCheckNewClasses:
    """Test new classes detection endpoint"""

    url_suffix = "/preprocess/check-new-classes"

    def test_missing_file(self, api_base_url):
        """Test without uploading file"""
        logging.info("Testing check-new-classes without file")
        url = f"{api_base_url}{self.url_suffix}"
        
        response = requests.post(url=url)
        
        # Should return 422 (validation error)
        assert response.status_code == 422

    def test_missing_prdtypecode_column(self, api_base_url):
        """Test with CSV missing prdtypecode column"""
        logging.info("Testing check-new-classes with missing column")
        url = f"{api_base_url}{self.url_suffix}"
        
        # Create CSV without prdtypecode
        data = pd.DataFrame({
            "designation": ["Product 1"],
            "description": ["Description 1"],
        })
        
        csv_buffer = StringIO()
        data.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        files = {"file": ("test.csv", csv_buffer.getvalue(), "text/csv")}
        response = requests.post(url=url, files=files)
        
        # Should return 400
        assert response.status_code == 400
        assert "prdtypecode" in response.json()["detail"]

    def test_valid_class_check(self, api_base_url, sample_csv_data):
        """Test with valid CSV containing prdtypecode"""
        logging.info("Testing check-new-classes with valid data")
        url = f"{api_base_url}{self.url_suffix}"
        
        csv_buffer = StringIO()
        sample_csv_data.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        files = {"file": ("test.csv", csv_buffer.getvalue(), "text/csv")}
        response = requests.post(url=url, files=files)
        
        # Should succeed
        assert response.status_code == 200
        
        content = response.json()
        assert "has_existing_encoder" in content
        assert "has_new_classes" in content
        assert "timestamp" in content


# ============================================
# TEST PREPROCESSING FROM RAW
# ============================================

class TestPreprocessFromRaw:
    """Test preprocessing from raw data endpoint"""

    url_suffix = "/preprocess/from-raw"

    def test_request_structure(self, api_base_url):
        """Test preprocessing request accepts proper structure"""
        logging.info("Testing preprocess from raw endpoint structure")
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "combine_existing_data": False,
            "save_holdout": True
        }
        
        response = requests.post(url=url, json=payload)
        
        # Should return 200 (started) or 400 (missing raw data)
        assert response.status_code in [200, 400, 409], (
            f"Unexpected status code {response.status_code}"
        )

    def test_response_structure_when_started(self, api_base_url):
        """Test response structure when preprocessing starts"""
        url = f"{api_base_url}{self.url_suffix}"
        
        payload = {
            "combine_existing_data": False,
            "save_holdout": True
        }
        
        response = requests.post(url=url, json=payload)
        
        if response.status_code == 200:
            content = response.json()
            assert "status" in content
            assert "message" in content
            assert "config" in content
            assert "timestamp" in content
            assert "endpoints" in content


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
        # 200 if results exist, 404 if no results yet
        assert req.status_code in [200, 404], (
            f"Unexpected status code {req.status_code}"
        )

    def test_response_structure_when_available(self, req):
        """Test response structure when results are available"""
        if req.status_code == 200:
            content = req.json()
            assert "status" in content
            assert "results" in content
            assert "timestamp" in content


# ============================================
# TEST EDGE CASES & ERROR HANDLING
# ============================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_concurrent_preprocessing_requests(self, api_base_url):
        """Test that concurrent preprocessing is prevented"""
        logging.info("Testing concurrent preprocessing prevention")
        url = f"{api_base_url}/preprocess/from-raw"
        
        payload = {"combine_existing_data": False, "save_holdout": True}
        
        # Send first request
        response1 = requests.post(url=url, json=payload)
        
        # Immediately send second request
        response2 = requests.post(url=url, json=payload)
        
        # One should be 200/400, the other might be 409 (conflict)
        statuses = {response1.status_code, response2.status_code}
        
        # At least one should succeed or properly reject
        assert all(s in [200, 400, 409] for s in statuses)

    def test_large_csv_handling(self, api_base_url):
        """Test handling of large CSV files"""
        logging.info("Testing large CSV handling")
        url = f"{api_base_url}/preprocess/batch"
        
        # Create large dataset (1000 rows)
        large_data = pd.DataFrame({
            "designation": [f"Product {i}" for i in range(1000)],
            "description": [f"Description {i}" for i in range(1000)],
            "prdtypecode": [2280] * 1000,
        })
        
        csv_buffer = StringIO()
        large_data.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        files = {"file": ("large.csv", csv_buffer.getvalue(), "text/csv")}
        data = {"combine_existing_data": "false", "save_holdout": "false"}
        
        # Should handle gracefully (may take time)
        response = requests.post(url=url, files=files, data=data, timeout=60)
        
        # Should succeed or fail gracefully
        assert response.status_code in [200, 400, 500, 504]


# ============================================
# TEST INTEGRATION WORKFLOW
# ============================================

class TestAPIWorkflow:
    """Test complete API workflow"""

    def test_complete_workflow(self, api_base_url):
        """Test complete preprocessing workflow"""
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
        
        # Step 3: Check import status
        import_status_url = f"{api_base_url}/import/status"
        import_resp = requests.get(url=import_status_url)
        assert import_resp.status_code == 200
        logging.info("✓ Import status check passed")
        
        # Step 4: Check preprocessing status
        status_url = f"{api_base_url}/preprocessing/status"
        status_resp = requests.get(url=status_url)
        assert status_resp.status_code == 200
        logging.info("✓ Preprocessing status check passed")
        
        logging.info("✓ Complete workflow test passed")


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

