"""
Simple tests for the FastAPI app. Run with: pytest
"""

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

valid_input = {
    "temperature": 30,
    "humidity": 80,
    "distance_km": 100,
    "product_type": "milk",
    "initial_shelf_life": 3,
    "packaging_quality": 1,
}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_accepts_valid_input():
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200


def test_predict_returns_prediction_field():
    response = client.post("/predict", json=valid_input)
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["High Spoilage Risk", "Low Spoilage Risk"]


def test_predict_returns_counterfactual_field():
    response = client.post("/predict", json=valid_input)
    data = response.json()
    assert "counterfactual" in data


def test_predict_returns_router_model_field():
    response = client.post("/predict", json=valid_input)
    data = response.json()
    assert data["router_model"] in ["hardy", "normal", "fragile"]


def test_predict_rejects_missing_field():
    bad_input = valid_input.copy()
    del bad_input["temperature"]
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422  # FastAPI validation error


def test_predict_rejects_wrong_type():
    bad_input = valid_input.copy()
    bad_input["temperature"] = "very hot"  # should be a number
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
