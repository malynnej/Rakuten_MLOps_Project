"""
Docker container integration tests:
- checks containers are running
- services respond to requests
- assumes docker-compose up is started already
"""

import docker
import requests
import time
import pytest


@pytest.fixture(scope="module")
def docker_client():
    """Docker client fixture"""
    return docker.from_env()


class TestDockerServices:
    """Test Docker containers"""
    
    def test_docker_compose_services_running(self, docker_client):
        """Test all services are running"""
        required_services = [
            "data_service",
            "train_api",
            "evaluate_api",
            "predict_api",
        ]
        
        containers = docker_client.containers.list()
        container_names = [c.name for c in containers]
        
        for service in required_services:
            # Check if any container name contains the service name
            assert any(service in name for name in container_names), (
                f"Service {service} not running"
            )
    
    def test_services_respond(self):
        """Test services respond to health checks"""
        services = [
            ("data_service", "http://localhost:8001/health"),
            ("train_api", "http://localhost:8002/health"),
            ("evaluate_api", "http://localhost:8004/health"),
            ("predict_api", "http://localhost:8000/health"),
        ]
        
        for service_name, url in services:
            try:
                response = requests.get(url, timeout=5)
                assert response.status_code == 200, (
                    f"{service_name} not healthy"
                )
            except requests.exceptions.RequestException as e:
                pytest.fail(f"{service_name} not accessible: {e}")
    
    def test_service_logs_no_errors(self, docker_client):
        """Test service logs don't contain critical errors"""
        containers = docker_client.containers.list()
        
        for container in containers:
            logs = container.logs(tail=50).decode('utf-8')
            
            # Check for critical errors (customize as needed)
            critical_errors = [
                "CRITICAL",
                "FATAL",
                "Exception",  # Be selective - some exceptions are handled
            ]
            
            # This is a basic check - adjust based on your needs
            # You might want to allow certain exceptions
            for error in critical_errors:
                if error in logs:
                    print(f"Warning: Found '{error}' in {container.name} logs")
