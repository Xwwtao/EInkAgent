"""HTTP API behavior tests."""

# pylint: disable=missing-function-docstring

from fastapi.testclient import TestClient

from eink_agent.api import app

CLIENT = TestClient(app)

def test_health_returns_service_status():
    response = CLIENT.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}