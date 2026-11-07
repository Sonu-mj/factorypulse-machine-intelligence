"""Unit and integration tests for FactoryPulse FastAPI prediction service.

Tests endpoints:
- GET /health
- GET /model/info
- POST /predict
- POST /predict/batch

Validates response schemas, HTTP status codes, error handling, input validation,
model unavailability guards, and prototype disclaimers.

Note on Dependency Requirements:
Starlette's TestClient requires 'httpx'. In accordance with project requirements to ask
before adding new dependencies, these tests invoke the FastAPI endpoint handlers directly
with Starlette Request/Response contracts, exercising complete routing logic, Pydantic
validation, error mapping, and status code behavior without requiring httpx.
"""

from unittest.mock import MagicMock
import pytest
from fastapi import HTTPException, Response
from pydantic import ValidationError
from starlette.requests import Request

from src.api.app import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    HealthResponse,
    ModelInfoResponse,
    app,
    create_app,
    get_predictor,
)
from src.prediction.predictor import FailurePredictor
from src.prediction.schemas import (
    APPROVED_FEATURE_NAMES,
    BatchPredictionResult,
    BatchTelemetryInput,
    MANDATORY_PROTOTYPE_DISCLAIMER,
    MachineTelemetryInput,
    PredictionResult,
)


@pytest.fixture
def test_app():
    """Create a fresh test FastAPI application."""
    return create_app()


@pytest.fixture
def predictor():
    """Default initialized FailurePredictor instance."""
    return FailurePredictor()


@pytest.fixture
def valid_telemetry():
    """Valid machine telemetry input."""
    return MachineTelemetryInput(
        machine_id="CNC-TEST-01",
        type="M",
        air_temperature_k=298.5,
        process_temperature_k=308.7,
        rotational_speed_rpm=1500,
        torque_nm=45.0,
        tool_wear_min=100,
    )


def build_starlette_request(fastapi_app, model_available: bool = True, predictor_instance=None):
    """Construct a mock Starlette Request object linked to the application state."""
    scope = {"type": "http", "app": fastapi_app}
    req = Request(scope)
    fastapi_app.state.model_available = model_available
    fastapi_app.state.predictor = predictor_instance
    return req


# -------------------------------------------------------------------------
# 1. Application Metadata & Config
# -------------------------------------------------------------------------

def test_application_metadata(test_app):
    """Verify application title, version, description, and prototype disclaimer."""
    assert test_app.title == API_TITLE
    assert test_app.version == API_VERSION
    assert "Industrial Machine Health & Failure-Risk Intelligence" in test_app.description
    assert MANDATORY_PROTOTYPE_DISCLAIMER in test_app.description


# -------------------------------------------------------------------------
# 2. GET /health
# -------------------------------------------------------------------------

def test_health_check_healthy(test_app, predictor):
    """Verify /health returns 200 and healthy status when model is operational."""
    req = build_starlette_request(test_app, model_available=True, predictor_instance=predictor)
    res = Response()

    health_route = next(r for r in test_app.routes if r.path == "/health")
    result = health_route.endpoint(request=req, response=res)

    assert res.status_code == 200
    assert isinstance(result, HealthResponse)
    assert result.status == "healthy"
    assert result.model_available is True
    assert result.version == API_VERSION
    assert result.disclaimer == MANDATORY_PROTOTYPE_DISCLAIMER


def test_health_check_unhealthy(test_app):
    """Verify /health returns 503 and unhealthy status when model is not available."""
    req = build_starlette_request(test_app, model_available=False, predictor_instance=None)
    res = Response()

    health_route = next(r for r in test_app.routes if r.path == "/health")
    result = health_route.endpoint(request=req, response=res)

    assert res.status_code == 503
    assert isinstance(result, HealthResponse)
    assert result.status == "unhealthy"
    assert result.model_available is False
    assert result.disclaimer == MANDATORY_PROTOTYPE_DISCLAIMER


# -------------------------------------------------------------------------
# 3. GET /model/info
# -------------------------------------------------------------------------

def test_get_model_info_success(test_app, predictor):
    """Verify /model/info retrieves metadata directly from FailurePredictor."""
    info_route = next(r for r in test_app.routes if r.path == "/model/info")
    result = info_route.endpoint(predictor=predictor)

    assert isinstance(result, ModelInfoResponse)
    assert result.model_name == "RandomForestClassifier (balanced)"
    assert "ColumnTransformer + RandomForestClassifier" in result.model_type
    assert result.selected_threshold == 0.45
    assert result.approved_features == APPROVED_FEATURE_NAMES
    assert result.disclaimer == MANDATORY_PROTOTYPE_DISCLAIMER


# -------------------------------------------------------------------------
# 4. POST /predict
# -------------------------------------------------------------------------

def test_predict_endpoint_success(test_app, predictor, valid_telemetry):
    """Verify /predict returns structured PredictionResult."""
    predict_route = next(r for r in test_app.routes if r.path == "/predict")
    result = predict_route.endpoint(telemetry=valid_telemetry, predictor=predictor)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.applied_threshold == 0.45
    assert result.predicted_class in [0, 1]
    assert result.prediction_label in ["Normal Operation", "Machine Failure"]
    assert result.machine_id == "CNC-TEST-01"
    assert result.disclaimer == MANDATORY_PROTOTYPE_DISCLAIMER


def test_predict_endpoint_model_unavailable(test_app):
    """Verify get_predictor raises 503 when model failed to initialize."""
    req = build_starlette_request(test_app, model_available=False, predictor_instance=None)

    with pytest.raises(HTTPException) as exc_info:
        get_predictor(req)

    assert exc_info.value.status_code == 503
    assert "currently unavailable" in exc_info.value.detail


def test_predict_endpoint_invalid_sensor_raises_validation_error():
    """Verify negative or physically impossible sensor values fail Pydantic validation (HTTP 422)."""
    with pytest.raises(ValidationError):
        MachineTelemetryInput(
            type="M",
            air_temperature_k=-273.15,
            process_temperature_k=308.7,
            rotational_speed_rpm=1500,
            torque_nm=45.0,
            tool_wear_min=100,
        )

    with pytest.raises(ValidationError):
        MachineTelemetryInput(
            type="INVALID_TYPE",
            air_temperature_k=298.5,
            process_temperature_k=308.7,
            rotational_speed_rpm=1500,
            torque_nm=45.0,
            tool_wear_min=100,
        )


def test_predict_endpoint_missing_fields_raises_validation_error():
    """Verify omitting required sensor fields fails Pydantic validation (HTTP 422)."""
    with pytest.raises(ValidationError):
        MachineTelemetryInput.model_validate({"type": "M", "air_temperature_k": 298.5})


def test_predict_endpoint_value_error_returns_400(test_app, valid_telemetry):
    """Verify ValueError from predictor logic translates to HTTP 400 Bad Request."""
    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = ValueError("Data leakage violation! Forbidden column detected.")

    predict_route = next(r for r in test_app.routes if r.path == "/predict")

    with pytest.raises(HTTPException) as exc_info:
        predict_route.endpoint(telemetry=valid_telemetry, predictor=mock_predictor)

    assert exc_info.value.status_code == 400
    assert "Data leakage violation" in exc_info.value.detail


def test_predict_endpoint_internal_error_hides_filesystem_paths(test_app, valid_telemetry):
    """Verify unexpected errors return HTTP 500 without leaking filesystem paths."""
    mock_predictor = MagicMock()
    mock_predictor.predict.side_effect = RuntimeError("Failed reading C:\\secret_path\\models\\weights.dat")

    predict_route = next(r for r in test_app.routes if r.path == "/predict")

    with pytest.raises(HTTPException) as exc_info:
        predict_route.endpoint(telemetry=valid_telemetry, predictor=mock_predictor)

    assert exc_info.value.status_code == 500
    assert "An internal error occurred" in exc_info.value.detail
    assert "secret_path" not in exc_info.value.detail


# -------------------------------------------------------------------------
# 5. POST /predict/batch
# -------------------------------------------------------------------------

def test_predict_batch_endpoint_success(test_app, predictor, valid_telemetry):
    """Verify /predict/batch generates aggregated BatchPredictionResult."""
    m2 = MachineTelemetryInput(
        machine_id="CNC-TEST-02",
        type="H",
        air_temperature_k=303.0,
        process_temperature_k=312.0,
        rotational_speed_rpm=1200,
        torque_nm=70.0,
        tool_wear_min=220,
    )
    batch = BatchTelemetryInput(machines=[valid_telemetry, m2])

    batch_route = next(r for r in test_app.routes if r.path == "/predict/batch")
    result = batch_route.endpoint(batch=batch, predictor=predictor)

    assert isinstance(result, BatchPredictionResult)
    assert result.total_evaluated == 2
    assert result.applied_threshold == 0.45
    assert 0 <= result.predicted_failures_count <= 2
    assert len(result.predictions) == 2
    assert result.predictions[0].machine_id == "CNC-TEST-01"
    assert result.predictions[1].machine_id == "CNC-TEST-02"
    assert result.disclaimer == MANDATORY_PROTOTYPE_DISCLAIMER


def test_predict_batch_empty_rejected_by_schema():
    """Verify empty batch is rejected at the schema level (min_length=1)."""
    with pytest.raises(ValidationError):
        BatchTelemetryInput(machines=[])


def test_predict_batch_empty_list_returns_400(test_app, predictor):
    """Verify route handler rejects empty machines list with HTTP 400."""
    batch_route = next(r for r in test_app.routes if r.path == "/predict/batch")

    mock_batch = MagicMock()
    mock_batch.machines = []

    with pytest.raises(HTTPException) as exc_info:
        batch_route.endpoint(batch=mock_batch, predictor=predictor)

    assert exc_info.value.status_code == 400
    assert "batch cannot be empty" in exc_info.value.detail
