"""Unit tests for the FactoryPulse Streamlit Dashboard API Client.

Validates payload construction, successful responses, connection errors, timeouts,
validation errors (422), internal errors (500), malformed responses, and health checks
without requiring a live FastAPI server.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from dashboard.api_client import FactoryPulseAPIClient


@pytest.fixture
def client():
    """Default initialized API client."""
    return FactoryPulseAPIClient(base_url="http://127.0.0.1:8000", timeout=3.0)


@pytest.fixture
def sample_telemetry():
    """Standard machine telemetry dictionary."""
    return {
        "machine_id": "CNC-01",
        "type": "M",
        "air_temperature_k": 298.5,
        "process_temperature_k": 308.7,
        "rotational_speed_rpm": 1500,
        "torque_nm": 45.0,
        "tool_wear_min": 100,
    }


# -------------------------------------------------------------------------
# 1. Health Check Tests
# -------------------------------------------------------------------------

@patch("requests.get")
def test_successful_health_check(mock_get, client):
    """Verify check_health returns success when backend is healthy and model is available."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "model_available": True,
        "version": "0.4.0",
        "disclaimer": "EXPERIMENTAL BENCHMARK PROTOTYPE",
    }
    mock_get.return_value = mock_resp

    res = client.check_health()

    assert res.success is True
    assert res.status_code == 200
    assert res.data["model_available"] is True
    assert res.error_message is None
    mock_get.assert_called_once_with("http://127.0.0.1:8000/health", timeout=2.5)


@patch("requests.get")
def test_failed_health_check_model_unavailable(mock_get, client):
    """Verify check_health returns failure when model_available is False (HTTP 503)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.json.return_value = {
        "status": "unhealthy",
        "model_available": False,
        "version": "0.4.0",
    }
    mock_get.return_value = mock_resp

    res = client.check_health()

    assert res.success is False
    assert res.status_code == 503
    assert "Backend unavailable" in res.error_message


@patch("requests.get")
def test_health_check_connection_failure(mock_get, client):
    """Verify check_health handles server down / connection refused."""
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

    res = client.check_health()

    assert res.success is False
    assert res.status_code is None
    assert "Disconnected" in res.error_message


@patch("requests.get")
def test_health_check_timeout(mock_get, client):
    """Verify check_health handles request timeout."""
    mock_get.side_effect = requests.exceptions.Timeout("Read timeout")

    res = client.check_health()

    assert res.success is False
    assert "timed out" in res.error_message


# -------------------------------------------------------------------------
# 2. Prediction Tests
# -------------------------------------------------------------------------

@patch("requests.post")
def test_successful_prediction_request(mock_post, client, sample_telemetry):
    """Verify predict returns success and parses valid prediction response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "failure_probability": 0.125,
        "applied_threshold": 0.45,
        "predicted_class": 0,
        "prediction_label": "Normal Operation",
        "machine_id": "CNC-01",
        "model_name": "RandomForestClassifier (balanced)",
        "disclaimer": "EXPERIMENTAL BENCHMARK PROTOTYPE",
    }
    mock_post.return_value = mock_resp

    res = client.predict(sample_telemetry)

    assert res.success is True
    assert res.status_code == 200
    assert res.data["failure_probability"] == 0.125
    assert res.data["applied_threshold"] == 0.45
    assert res.data["predicted_class"] == 0
    assert res.data["prediction_label"] == "Normal Operation"
    assert res.data["machine_id"] == "CNC-01"


@patch("requests.post")
def test_correct_json_payload_sent(mock_post, client, sample_telemetry):
    """Verify predict sends the exact telemetry dictionary as JSON with correct URL and timeout."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "failure_probability": 0.65,
        "applied_threshold": 0.45,
        "predicted_class": 1,
        "prediction_label": "Machine Failure",
    }
    mock_post.return_value = mock_resp

    client.predict(sample_telemetry)

    mock_post.assert_called_once_with(
        "http://127.0.0.1:8000/predict",
        json=sample_telemetry,
        timeout=3.0,
    )


@patch("requests.post")
def test_prediction_backend_connection_failure(mock_post, client, sample_telemetry):
    """Verify predict returns clear error message on ConnectionError."""
    mock_post.side_effect = requests.exceptions.ConnectionError("Failed to establish a new connection")

    res = client.predict(sample_telemetry)

    assert res.success is False
    assert "Connection Refused" in res.error_message
    assert "Ensure the FastAPI service is running" in res.error_message


@patch("requests.post")
def test_prediction_request_timeout(mock_post, client, sample_telemetry):
    """Verify predict returns descriptive message on Timeout."""
    mock_post.side_effect = requests.exceptions.Timeout("Request timed out after 3.0s")

    res = client.predict(sample_telemetry)

    assert res.success is False
    assert "Timed Out" in res.error_message


@patch("requests.post")
def test_prediction_http_422_validation_error(mock_post, client, sample_telemetry):
    """Verify predict formats Pydantic HTTP 422 error details cleanly."""
    mock_resp = MagicMock()
    mock_resp.status_code = 422
    mock_resp.json.return_value = {
        "detail": [
            {"loc": ["body", "air_temperature_k"], "msg": "Input should be greater than 0", "type": "greater_than"}
        ]
    }
    mock_post.return_value = mock_resp

    res = client.predict(sample_telemetry)

    assert res.success is False
    assert res.status_code == 422
    assert "Validation Error (HTTP 422)" in res.error_message
    assert "air_temperature_k" in res.error_message


@patch("requests.post")
def test_prediction_http_500_internal_error(mock_post, client, sample_telemetry):
    """Verify predict handles HTTP 500 without crashing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {
        "detail": "An internal error occurred while generating the prediction."
    }
    mock_post.return_value = mock_resp

    res = client.predict(sample_telemetry)

    assert res.success is False
    assert res.status_code == 500
    assert "API Server Error (HTTP 500)" in res.error_message


@patch("requests.post")
def test_prediction_malformed_non_json_response(mock_post, client, sample_telemetry):
    """Verify predict handles non-JSON responses from server gracefully."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = ValueError("Invalid JSON")
    mock_post.return_value = mock_resp

    res = client.predict(sample_telemetry)

    assert res.success is False
    assert "malformed, non-JSON response" in res.error_message


@patch("requests.post")
def test_prediction_malformed_missing_fields_response(mock_post, client, sample_telemetry):
    """Verify predict rejects 200 responses that lack required prediction attributes."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"only_some_unrelated_field": 123}
    mock_post.return_value = mock_resp

    res = client.predict(sample_telemetry)

    assert res.success is False
    assert "missing required fields" in res.error_message


def test_prediction_empty_telemetry_rejected(client):
    """Verify predict immediately rejects empty input dict without making network call."""
    res = client.predict({})
    assert res.success is False
    assert "non-empty dictionary" in res.error_message

