"""HTTP API behavior tests."""

# pylint: disable=missing-function-docstring

from fastapi.testclient import TestClient

from eink_agent.api import app

CLIENT = TestClient(app)

def test_health_returns_service_status():
    response = CLIENT.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_search_devices_supports_query_parameters():
    response = CLIENT.get(
        "/devices",
        params={
            "brand": "DemoInk",
            "max_price": 2000,
            "limit": 10,
        },
    )

    assert response.status_code == 200
    devices = response.json()
    assert len(devices) == 1
    assert devices[0]["brand"] == "DemoInk"
    assert devices[0]["model"] == "Reader 6"


def test_search_devices_rejects_negative_max_price():
    response = CLIENT.get("/devices", params={"max_price": -1})

    assert response.status_code == 422


def test_search_devices_rejects_invalid_limit():
    response = CLIENT.get("/devices", params={"limit": 0})

    assert response.status_code == 422

def test_get_device_detail_returns_requested_device():
    response = CLIENT.get("/devices/1")

    assert response.status_code == 200
    device = response.json()
    assert device["id"] == 1
    assert device["model"] == "Reader 6"

def test_get_device_detail_returns_404_for_unknown_device():
    response = CLIENT.get("/devices/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Device not found"}

def test_get_device_detail_rejects_invalid_device_id():
    response = CLIENT.get("/devices/0")

    assert response.status_code == 422