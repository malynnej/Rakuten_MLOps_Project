import os
from datetime import datetime
from typing import List
import requests
import pytest
import logging


# definition of the API address and port
API_ADDRESS = os.environ.get('API_ADDRESS', default='127.0.0.1')
API_PORT = os.environ.get('API_PORT', default='8000')


class TestHealth:
    """Test class for health API endpoint"""

    @pytest.fixture(scope='class')
    def req(self):
        """Request on the health endpoint"""
        logging.info("Sending request to health endpoint")
        url = f'http://{API_ADDRESS}:{API_PORT}/health'
        response = requests.get(url=url)
        return response
    
    @pytest.fixture(scope='class')
    def req_content(self, req):
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
    def test_health_types(self, req_content, key, expected_type):
        """Test the type of the health endpoint response content"""
        assert isinstance(req_content[key], expected_type), \
            f"Content type for '{key}' is not {expected_type}"

    def test_health_timestamp_format(self, req_content):
        """Test the timestamp format of the health endpoint response"""
        try:
            datetime.fromisoformat(req_content['timestamp'])
        except ValueError:
            assert False, "Timestamp is not in valid ISO format"

    def test_health_status_message(self, req_content):
        """Test the status message of the health endpoint response"""
        expected_status_str = 'healthy'
        assert req_content['status'] == expected_status_str, \
            f"Expected status '{expected_status_str}', got '{req_content['status']}'"


class TestModelInfo:
    """Test class for model info API endpoint"""

    @pytest.fixture(scope='class')
    def req(self):
        """Request on the model info endpoint"""
        logging.info("Sending request to model info endpoint")
        url = f'http://{API_ADDRESS}:{API_PORT}/model/info'
        response = requests.get(url=url)
        return response
    
    @pytest.fixture(scope='class')
    def req_content(self, req):
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
    def test_types(self, req_content, key, expected_type):
        """Test the type of the model info endpoint response content"""
        assert isinstance(req_content[key], expected_type), \
            f"Content type for '{key}' is not {expected_type}"


class TestPredictSingle:
    """Test class for single prediction API endpoint"""
    test_text = "Zombie action figure collectible model"

    @pytest.fixture(scope='class')
    def req(self):
        """Request on the single prediction endpoint"""
        logging.info("Sending request to single prediction endpoint")
        url = f'http://{API_ADDRESS}:{API_PORT}/predict'
        payload = {
            "text": self.test_text
        }
        response = requests.post(url=url, json=payload)
        return response
    
    @pytest.fixture(scope='class')
    def req_content(self, req):
        """Extract content from the single prediction endpoint response"""
        logging.info("Extracting content from single prediction endpoint response")
        return req.json()

    def test_status_code(self, req):
        """Test the status code of the single prediction endpoint"""
        logging.info("Testing single prediction endpoint status code")
        expected_status_code = 200
        assert req.status_code == expected_status_code, \
            f"Expected status code {expected_status_code}, got {req.status_code}"
    
    @pytest.mark.parametrize("key, expected_type", [
        ("text", str),
        ("predicted_category", int),
        ("confidence", float),
        ("predicted_index", int),
    ])
    def test_keys_types(self, req_content, key, expected_type):
        """Test the returned content keys and types"""
        assert key in req_content, \
            f"Key '{key}' not found in response"
        assert isinstance(req_content[key], expected_type), \
            f"Content type for '{key}' is not {expected_type}"

    def test_input_text_match(self, req_content):
        """Test that the input text matches the response"""
        expected_text = self.test_text
        assert req_content['text'] == expected_text, \
            f"Expected text '{expected_text}', got '{req_content['text']}'"


class TestPredictBatch:
    """Test class for batch prediction API endpoint"""
    test_texts = [
        "Wireless Bluetooth headphones with noise cancellation",
        "Organic green tea bags for healthy lifestyle",
        "Smartphone with high-resolution camera and long battery life"
    ]

    @pytest.fixture(scope='class')
    def req(self):
        """Request on the batch prediction endpoint"""
        logging.info("Sending request to batch prediction endpoint")
        url = f'http://{API_ADDRESS}:{API_PORT}/predict/batch'
        payload = {
            "texts": self.test_texts
        }
        response = requests.post(url=url, json=payload)
        return response

    @pytest.fixture(scope='class')
    def req_content(self, req):
        """Extract content from the batch prediction endpoint response"""
        logging.info("Extracting content from batch prediction endpoint response")
        return req.json()

    def test_status_code(self, req):
        """Test the status code of the batch prediction endpoint"""
        logging.info("Testing batch prediction endpoint status code")
        expected_status_code = 200
        assert req.status_code == expected_status_code, \
            f"Expected status code {expected_status_code}, got {req.status_code}"

    @pytest.mark.parametrize("key, expected_type", [
        ("count", int),
        ("predictions", List),
    ])
    def test_response_structure_types(self, req_content, key, expected_type):
        """Test the structure of the batch prediction response"""
        assert key in req_content, \
            f"Key '{key}' not found in response"
        assert isinstance(req_content[key], expected_type), \
            f"Content type for '{key}' is not {expected_type}"

    def test_predictions_length(self, req_content):
        """Test that the returned predictions length matches the input list length"""
        expected_length = len(self.test_texts)
        assert req_content['count'] == expected_length, \
            f"Expected count {expected_length}, got {req_content['count']}"
        assert len(req_content['predictions']) == expected_length, \
            f"Expected predictions length {expected_length}, got {len(req_content['predictions'])}"

    def test_input_text_match(self, req_content):
        """Test that the input text matches the response"""
        expected_texts = self.test_texts
        returned_texts = [item['text'] for item in req_content['predictions']]
        assert returned_texts == expected_texts, \
            f"Expected texts {expected_texts}, got {returned_texts}"
