"""Unit and integration tests for FactoryPulse FailurePredictor service.

Validates model artifact loading, threshold resolution, fallback behavior, exact boundary logic,
anti-leakage restrictions, input validation, and vectorized batch inference.
"""

import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.prediction.predictor import (
    DEFAULT_FALLBACK_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD_PATH,
    FailurePredictor,
)
from src.prediction.schemas import (
    APPROVED_FEATURE_NAMES,
    BatchTelemetryInput,
    FORBIDDEN_COLUMNS,
    MANDATORY_PROTOTYPE_DISCLAIMER,
    MachineTelemetryInput,
    PredictionResult,
)


@pytest.fixture
def valid_telemetry_dict():
    """Valid standard operating machine telemetry."""
    return {
        "machine_id": "CNC-UNIT-01",
        "type": "M",
        "air_temperature_k": 298.5,
        "process_temperature_k": 308.7,
        "rotational_speed_rpm": 1500,
        "torque_nm": 42.0,
        "tool_wear_min": 85,
    }


@pytest.fixture
def valid_telemetry_input(valid_telemetry_dict):
    """Pydantic model instance of valid telemetry."""
    return MachineTelemetryInput(**valid_telemetry_dict)


@pytest.fixture
def predictor():
    """Default initialized FailurePredictor instance."""
    return FailurePredictor(
        model_path=DEFAULT_MODEL_PATH,
        threshold_path=DEFAULT_THRESHOLD_PATH,
    )


# -------------------------------------------------------------------------
# 1. Model Loading & Verification
# -------------------------------------------------------------------------

def test_successful_model_loading(predictor):
    """Verify default model artifact loads successfully as an sklearn Pipeline."""
    assert predictor.pipeline is not None
    assert isinstance(predictor.pipeline, Pipeline)
    assert hasattr(predictor.pipeline, "predict_proba")
    assert predictor.model_path.exists()


def test_missing_model_artifact_raises_file_not_found(tmp_path):
    """Verify FileNotFoundError is raised when model artifact is absent."""
    missing_path = tmp_path / "nonexistent_model.joblib"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        FailurePredictor(model_path=missing_path)


def test_malformed_model_artifact_raises_value_error(tmp_path):
    """Verify ValueError is raised when model artifact file is corrupt."""
    corrupt_path = tmp_path / "corrupt_model.joblib"
    corrupt_path.write_bytes(b"NOT_A_VALID_JOBLIB_FILE")

    with pytest.raises(ValueError, match="Failed to load model artifact"):
        FailurePredictor(model_path=corrupt_path)


def test_invalid_model_type_raises_type_error(tmp_path):
    """Verify TypeError is raised if loaded object lacks predict_proba."""
    invalid_obj = {"not": "a_model"}
    invalid_path = tmp_path / "not_pipeline.joblib"
    joblib.dump(invalid_obj, invalid_path)

    with pytest.raises(TypeError, match="missing predict_proba"):
        FailurePredictor(model_path=invalid_path)


# -------------------------------------------------------------------------
# 2. Threshold Loading & Fallback
# -------------------------------------------------------------------------

def test_correct_threshold_loading_from_report(predictor):
    """Verify recommended threshold 0.45 is loaded from the threshold report."""
    assert predictor.threshold == 0.45
    assert isinstance(predictor.threshold, float)


def test_custom_threshold_loading(tmp_path):
    """Verify threshold report correctly parses custom threshold."""
    custom_report = tmp_path / "custom_threshold.json"
    custom_report.write_text(
        json.dumps({"recommendation": {"recommended_threshold": 0.38}}),
        encoding="utf-8",
    )

    pred = FailurePredictor(threshold_path=custom_report)
    assert pred.threshold == 0.38


def test_fallback_threshold_when_report_missing(tmp_path, caplog):
    """Verify fallback threshold 0.45 is used with a warning when report is missing."""
    missing_report = tmp_path / "missing_report.json"

    with caplog.at_level(logging.WARNING):
        pred = FailurePredictor(threshold_path=missing_report, fallback_threshold=0.45)

    assert pred.threshold == 0.45
    assert any("Threshold report not found" in record.message for record in caplog.records)
    assert any("Using documented fallback threshold: 0.45" in record.message for record in caplog.records)


def test_fallback_threshold_when_report_is_none(caplog):
    """Verify fallback threshold 0.45 is used when threshold_path is None."""
    with caplog.at_level(logging.WARNING):
        pred = FailurePredictor(threshold_path=None, fallback_threshold=0.45)

    assert pred.threshold == 0.45
    assert any("Threshold report path is None" in record.message for record in caplog.records)


def test_malformed_threshold_report_invalid_json(tmp_path):
    """Verify ValueError is raised if threshold report has syntax errors."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad_json: True,", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON syntax"):
        FailurePredictor(threshold_path=bad_json)


def test_malformed_threshold_report_missing_recommendation_key(tmp_path):
    """Verify ValueError is raised if recommendation key is missing."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"wrong_key": 0.45}), encoding="utf-8")

    with pytest.raises(ValueError, match="Missing 'recommendation.recommended_threshold'"):
        FailurePredictor(threshold_path=bad_json)


def test_malformed_threshold_report_invalid_value(tmp_path):
    """Verify ValueError is raised if threshold is not a valid number in [0, 1]."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(
        json.dumps({"recommendation": {"recommended_threshold": 1.75}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        FailurePredictor(threshold_path=bad_json)


# -------------------------------------------------------------------------
# 3. Probability Calculation & Prediction Output
# -------------------------------------------------------------------------

def test_probability_calculation_and_predict_single(predictor, valid_telemetry_input):
    """Verify single prediction computes valid probability, threshold, and label."""
    result = predictor.predict(valid_telemetry_input)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.applied_threshold == 0.45
    assert result.predicted_class in [0, 1]
    assert result.prediction_label in ["Normal Operation", "Machine Failure"]
    assert result.machine_id == "CNC-UNIT-01"
    assert result.model_name == "RandomForestClassifier (balanced)"
    assert result.disclaimer == MANDATORY_PROTOTYPE_DISCLAIMER


def test_predict_with_raw_dictionary_and_column_names(predictor):
    """Verify prediction accepts dictionary with original dataset column names."""
    raw_dict = {
        "Type": "L",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1400,
        "Torque [Nm]": 50.0,
        "Tool wear [min]": 200,
    }
    result = predictor.predict(raw_dict)
    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.failure_probability <= 1.0
    assert result.machine_id is None


# -------------------------------------------------------------------------
# 4. Exact Decision Rule & Boundary Behavior
# -------------------------------------------------------------------------

def test_exact_threshold_boundary_rule(predictor):
    """Verify explicit decision rule: probability >= threshold."""
    # Test boundary logic using synthetic mocked probability values
    threshold = 0.45

    class MockPipeline:
        def __init__(self, proba_val):
            self.proba_val = proba_val

        def predict_proba(self, X):
            return np.array([[1.0 - self.proba_val, self.proba_val]])

    # Exactly below threshold: 0.449999 -> Normal Operation (0)
    predictor.pipeline = MockPipeline(0.449999)
    res_below = predictor.predict({
        "type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100,
    })
    assert res_below.predicted_class == 0
    assert res_below.prediction_label == "Normal Operation"

    # Exactly AT threshold: 0.450000 -> Machine Failure (1)
    predictor.pipeline = MockPipeline(0.450000)
    res_at = predictor.predict({
        "type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100,
    })
    assert res_at.predicted_class == 1
    assert res_at.prediction_label == "Machine Failure"

    # Above threshold: 0.450001 -> Machine Failure (1)
    predictor.pipeline = MockPipeline(0.450001)
    res_above = predictor.predict({
        "type": "L", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100,
    })
    assert res_above.predicted_class == 1
    assert res_above.prediction_label == "Machine Failure"


# -------------------------------------------------------------------------
# 5. Anti-Leakage & Input Validation
# -------------------------------------------------------------------------

@pytest.mark.parametrize("forbidden_col", FORBIDDEN_COLUMNS)
def test_rejection_of_forbidden_columns(predictor, valid_telemetry_dict, forbidden_col):
    """Verify inputs containing identifier or failure-mode columns are strictly rejected."""
    leaked_input = valid_telemetry_dict.copy()
    leaked_input[forbidden_col] = 1

    with pytest.raises(ValueError, match="Data leakage violation"):
        predictor.predict(leaked_input)


def test_rejection_of_invalid_type_category(predictor, valid_telemetry_dict):
    """Verify unknown machine variant category is rejected."""
    bad_input = valid_telemetry_dict.copy()
    bad_input["type"] = "INVALID_TYPE"

    with pytest.raises(ValueError, match="Telemetry validation error"):
        predictor.predict(bad_input)


@pytest.mark.parametrize("param, bad_val", [
    ("air_temperature_k", -10.0),
    ("air_temperature_k", 0.0),
    ("process_temperature_k", -5.0),
    ("rotational_speed_rpm", -100),
    ("rotational_speed_rpm", 0),
    ("torque_nm", -1.0),
    ("tool_wear_min", -10),
])
def test_rejection_of_negative_and_zero_invalid_sensors(predictor, valid_telemetry_dict, param, bad_val):
    """Verify physical sensor constraint violations are rejected."""
    bad_input = valid_telemetry_dict.copy()
    bad_input[param] = bad_val

    with pytest.raises(ValueError, match="Telemetry validation error"):
        predictor.predict(bad_input)


def test_rejection_of_missing_features(predictor):
    """Verify error when required features are missing."""
    incomplete = {"type": "L", "air_temperature_k": 300.0}

    with pytest.raises(ValueError, match="Telemetry validation error"):
        predictor.predict(incomplete)


def test_rejection_of_invalid_input_types(predictor):
    """Verify non-dict, non-Pydantic input types raise ValueError."""
    with pytest.raises(ValueError, match="Invalid telemetry input type"):
        predictor.predict("not_a_dictionary")  # type: ignore

    with pytest.raises(ValueError, match="Invalid telemetry input type"):
        predictor.predict([1, 2, 3])  # type: ignore


# -------------------------------------------------------------------------
# 6. Vectorized Batch Prediction
# -------------------------------------------------------------------------

def test_predict_batch_success(predictor, valid_telemetry_input):
    """Verify batch prediction processes multiple inputs and preserves metadata."""
    m1 = valid_telemetry_input
    m2 = MachineTelemetryInput(
        machine_id="CNC-UNIT-02",
        type="H",
        air_temperature_k=304.0,
        process_temperature_k=313.0,
        rotational_speed_rpm=1300,
        torque_nm=65.0,
        tool_wear_min=240,
    )
    batch = [m1, m2]

    results = predictor.predict_batch(batch)

    assert len(results) == 2
    assert results[0].machine_id == "CNC-UNIT-01"
    assert results[1].machine_id == "CNC-UNIT-02"
    assert all(isinstance(r, PredictionResult) for r in results)
    assert all(r.applied_threshold == 0.45 for r in results)


def test_predict_batch_via_container_schema(predictor, valid_telemetry_input):
    """Verify predict_batch handles BatchTelemetryInput schema container."""
    container = BatchTelemetryInput(machines=[valid_telemetry_input])
    results = predictor.predict_batch(container)

    assert len(results) == 1
    assert results[0].machine_id == "CNC-UNIT-01"


def test_predict_batch_empty_raises_value_error(predictor):
    """Verify empty batch input is rejected."""
    with pytest.raises(ValueError, match="batch cannot be empty"):
        predictor.predict_batch([])


# -------------------------------------------------------------------------
# 7. Absence of Arbitrary Risk Tiers & Presence of Disclaimers
# -------------------------------------------------------------------------

def test_no_arbitrary_risk_tiers_in_prediction_result(predictor, valid_telemetry_input):
    """Verify PredictionResult does not invent unapproved risk-tier boundaries."""
    result = predictor.predict(valid_telemetry_input)
    result_dict = result.to_dict()

    # Confirm arbitrary risk tier fields are not present
    assert "risk_level" not in result_dict
    assert "risk_tier" not in result_dict
    assert "risk_factors" not in result_dict

    # Confirm required fields exist
    assert "failure_probability" in result_dict
    assert "applied_threshold" in result_dict
    assert "predicted_class" in result_dict
    assert "prediction_label" in result_dict
    assert "machine_id" in result_dict


def test_service_info_metadata(predictor):
    """Verify get_service_info reports correct metadata and features."""
    info = predictor.get_service_info()
    assert info["applied_threshold"] == 0.45
    assert info["approved_features"] == APPROVED_FEATURE_NAMES
    assert info["disclaimer"] == MANDATORY_PROTOTYPE_DISCLAIMER

