import os
from datetime import datetime
import requests
import pytest
import logging


# definition of the API address and port
API_ADDRESS = os.environ.get('API_ADDRESS', default='127.0.0.1')
API_PORT = os.environ.get('API_PORT', default='8000')


class TestHealth:
    """Test class for health API endpoint"""

    @pytest.fixture
    def req(self):
        """Request on the health endpoint"""
        logging.info("Sending request to health endpoint")
        url = f'http://{API_ADDRESS}:{API_PORT}/health'
        response = requests.get(url=url)
        return response
    
    @pytest.fixture
    def request_content(self, req):
        """Extract content from the health endpoint response"""
        logging.info("Extracting content from health endpoint response")
        return req.json()
    
    def test_health_status_code(self, req):
        """Test the status code of the health endpoint"""
        logging.info("Testing health endpoint status code")
        expected_status_code = 200
        assert req.status_code == expected_status_code, \
            f"Expected status code {expected_status_code}, got {req.status_code}"
    
    @pytest.mark.parametrize("key, expected_type", [
        ("status", str),
        ("current_model", str),
        ("timestamp", str)
    ])
    def test_health_types(self, request_content, key, expected_type):
        """Test the type of the health endpoint response content"""
        assert isinstance(request_content[key], expected_type), \
            f"Content type for '{key}' is not {expected_type}"

    def test_health_timestamp_format(self, request_content):
        """Test the timestamp format of the health endpoint response"""
        try:
            datetime.fromisoformat(request_content['timestamp'])
        except ValueError:
            assert False, "Timestamp is not in valid ISO format"

    def test_health_status_message(self, request_content):
        """Test the status message of the health endpoint response"""
        expected_status_str = 'healthy'
        assert request_content['status'] == expected_status_str, \
            f"Expected status '{expected_status_str}', got '{request_content['status']}'"


class TestModelInfo:
    """Test class for model info API endpoint"""

    @pytest.fixture
    def req(self):
        """Request on the model info endpoint"""
        logging.info("Sending request to model info endpoint")
        url = f'http://{API_ADDRESS}:{API_PORT}/model/info'
        response = requests.get(url=url)
        return response
    
    @pytest.fixture
    def request_content(self, req):
        """Extract content from the model info endpoint response"""
        logging.info("Extracting content from model info endpoint response")
        return req.json()

    def test_status_code(self, req):
        """Test the status code of the model info endpoint"""
        logging.info("Testing model info endpoint status code")
        expected_status_code = 200
        assert req.status_code == expected_status_code, \
            f"Expected status code {expected_status_code}, got {req.status_code}"


    @pytest.mark.parametrize("key, expected_type", [
        ("model_path", str),
        ("num_classes", int),
        ("device", str)
    ])
    def test_types(self, request_content, key, expected_type):
        """Test the type of the model info endpoint response content"""
        assert isinstance(request_content[key], expected_type), \
            f"Content type for '{key}' is not {expected_type}"
