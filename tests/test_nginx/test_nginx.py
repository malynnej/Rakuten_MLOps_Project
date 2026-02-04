import logging
import os
import time

import pytest
import requests
from requests.auth import HTTPBasicAuth

# definition of the API address and port
NGINX_ADDRESS = os.environ.get("NGINX_ADDRESS", default="127.0.0.1")
HTTPS_PORT = os.environ.get("HTTPS_PORT", default="8443")
HTTP_PORT = os.environ.get("HTTP_PORT", default="8080")

# waiting time for rate limited endpoints
SLEEP_SECS = .5

class TestHealthCheck:
    """Test class for Nginx health check endpoint"""

    expected_status_code = 200
    expected_text = "OK\n"

    @pytest.fixture(scope="class")
    def req(self):
        """Request on the health check endpoint"""
        logging.info("Sending request to health check endpoint")
        url = f"http://{NGINX_ADDRESS}:{HTTP_PORT}/health"
        response = requests.get(url=url)
        return response

    def test_health_status_code(self, req):
        """Test the status code of the health check response"""
        logging.info("Testing health check status code")
        assert req.status_code == self.expected_status_code, (
            f"Expected status code {self.expected_status_code}, got {req.status_code}"
        )

    def test_health_response_text(self, req):
        """Test the text content of the health check response"""
        logging.info("Testing health check response text")
        assert req.text == self.expected_text, (
            f"Expected text '{self.expected_text}', got '{req.text}'"
        )


class TestRedirect:
    """Test class for HTTP to HTTPS redirection"""

    @pytest.fixture(scope="class")
    def req(self):
        """Request on the HTTP endpoint"""
        logging.info("Sending request to HTTP endpoint for redirection test")
        url = f"http://{NGINX_ADDRESS}:{HTTP_PORT}/"
        response = requests.get(url=url, allow_redirects=False)
        return response

    def test_redirect_status_code(self, req):
        """Test the status code for redirection"""
        logging.info("Testing redirection status code")
        expected_status_code = 301  # Moved Permanently
        assert req.status_code == expected_status_code, (
            f"Expected redirect status code {expected_status_code}, got {req.status_code}"
        )


class TestNginxStatus:
    """Test access to nginx status endpoint"""

    url = f"https://{NGINX_ADDRESS}:{HTTPS_PORT}/nginx_status"

    def test_request_unauthorized(self):
        """Test response to unauthorized request"""
        expected_status_code = 401
        response = requests.get(url=self.url)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_request_invalid(self):
        """Test response to request with invalid credentials"""
        expected_status_code = 401
        response = requests.get(url=self.url, auth=HTTPBasicAuth("admin1", "admin2"))
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_request_valid(self):
        """Test response to request with valid credentials"""
        expected_status_code = 200
        response = requests.get(url=self.url, auth=HTTPBasicAuth("admin1", "admin1"))
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )


class TestForwardPredict:
    """Test forwarding to predict API"""

    baseUrl = f"https://{NGINX_ADDRESS}:{HTTPS_PORT}/predict"
    auth = HTTPBasicAuth("user1", "user1")

    @pytest.fixture(autouse=True, scope="class")
    def slow_down(self):
        """Slow down API tests to not reach request limit"""
        yield
        logging.info(f"Finished {self}")
        time.sleep(SLEEP_SECS)

    def test_root_unauthorized(self):
        """Test response to unauthorized request"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_invalid(self):
        """Test response to request with invalid credentials"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=HTTPBasicAuth("user1", "user2"))
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_valid(self):
        """Test response on the predict/ endpoint"""
        expected_status_code = 200
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=self.auth)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )


class TestForwardData:
    """Test forwarding to data API"""

    baseUrl = f"https://{NGINX_ADDRESS}:{HTTPS_PORT}/data"
    auth = HTTPBasicAuth("dev1", "dev1")

    @pytest.fixture(autouse=True, scope="class")
    def slow_down(self):
        """Slow down API tests to not reach request limit"""
        yield
        logging.info(f"Finished {self}")
        time.sleep(SLEEP_SECS)

    def test_root_unauthorized(self):
        """Test response to unauthorized request"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_invalid(self):
        """Test response to request with invalid credentials"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=HTTPBasicAuth("user1", "user1"))
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_valid(self):
        """Test response on the data/ endpoint"""
        expected_status_code = 200
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=self.auth)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )


class TestForwardTrain:
    """Test forwarding to train API"""

    baseUrl = f"https://{NGINX_ADDRESS}:{HTTPS_PORT}/train"
    auth = HTTPBasicAuth("dev2", "dev2")

    @pytest.fixture(autouse=True, scope="class")
    def slow_down(self):
        """Slow down API tests to not reach request limit"""
        yield
        logging.info(f"Finished {self}")
        time.sleep(SLEEP_SECS)

    def test_root_unauthorized(self):
        """Test response to unauthorized request"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_invalid(self):
        """Test response to request with invalid credentials"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=HTTPBasicAuth("dev2", "dev1"))
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_valid(self):
        """Test response on the train/ endpoint"""
        expected_status_code = 200
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=self.auth)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )


class TestForwardEvaluate:
    """Test forwarding to evaluate API"""

    baseUrl = f"https://{NGINX_ADDRESS}:{HTTPS_PORT}/evaluate"
    auth = HTTPBasicAuth("dev2", "dev2")

    @pytest.fixture(autouse=True, scope="class")
    def slow_down(self):
        """Slow down API tests to not reach request limit"""
        yield
        logging.info(f"Finished {self}")
        time.sleep(SLEEP_SECS)

    def test_root_unauthorized(self):
        """Test response to unauthorized request"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_invalid(self):
        """Test response to request with invalid credentials"""
        expected_status_code = 401
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=HTTPBasicAuth("dev2", "dev1"))
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )

    def test_root_valid(self):
        """Test response on the evaluate/ endpoint"""
        expected_status_code = 200
        url = f"{self.baseUrl}/"
        response = requests.get(url=url, auth=self.auth)
        assert response.status_code == expected_status_code, (
            f"Expected {expected_status_code}, got {response.status_code}"
        )
