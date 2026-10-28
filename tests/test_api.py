import pytest

from fastapi.testclient import TestClient
from src.api.main import app


# Create test client with model loading
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# Test health endpoint
def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# Test prediction endpoint
def test_prediction_endpoint(client):

    payload = {
        "review_text": "I really love this dress."
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["label"] == "Recommended"
    assert data["threshold"] == pytest.approx(0.40)

    assert 0 <= data["probability_recommended"] <= 1
    assert 0 <= data["probability_not_recommended"] <= 1

    assert (
        data["probability_recommended"]
        + data["probability_not_recommended"]
    ) == pytest.approx(1.0, abs=0.0002)


# Test empty review
def test_empty_review(client):

    response = client.post(
        "/predict",
        json={"review_text": ""}
    )

    assert response.status_code == 422


# Test whitespace-only review
def test_whitespace_review(client):

    response = client.post(
        "/predict",
        json={"review_text": "   "}
    )

    assert response.status_code == 422


# Test missing review field
def test_missing_review_field(client):

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


# Test review length limit
def test_review_too_long(client):

    response = client.post(
        "/predict",
        json={"review_text": "A" * 5001}
    )

    assert response.status_code == 422