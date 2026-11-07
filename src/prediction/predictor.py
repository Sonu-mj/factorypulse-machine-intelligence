"""FactoryPulse prediction service for industrial equipment failure risk assessment.

Loads the approved Random Forest pipeline artifact and recommended classification threshold,
validates incoming telemetry against strict anti-leakage contracts, and executes single
or batch inference directly through the embedded sklearn Pipeline without manual preprocessing.

EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union
import joblib
import pandas as pd
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from src.prediction.schemas import (
    APPROVED_FEATURE_NAMES,
    BatchTelemetryInput,
    FORBIDDEN_COLUMNS,
    MANDATORY_PROTOTYPE_DISCLAIMER,
    MachineTelemetryInput,
    PredictionResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH: Final[str] = "models/random_forest_pipeline.joblib"
DEFAULT_THRESHOLD_PATH: Final[str] = "reports/rf_threshold_analysis.json"
DEFAULT_FALLBACK_THRESHOLD: Final[float] = 0.45

# Column name mapping from dataset headers to schema fields
COLUMN_TO_FIELD_MAP: Final[Dict[str, str]] = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}


class FailurePredictor:
    """Service class for machine failure prediction.

    Loads the fitted Random Forest pipeline artifact and the decision threshold,
    evaluating incoming telemetry according to the explicit decision rule:
        is_failure = (probability >= threshold)

    Disclaimers:
        This service is an experimental benchmark prototype and has not been validated
        for operational industrial deployment. Arbitrary risk tiers (LOW/MEDIUM/HIGH/CRITICAL)
        are intentionally omitted as domain-approved risk tier definitions do not exist.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        threshold_path: Optional[Union[str, Path]] = DEFAULT_THRESHOLD_PATH,
        fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    ):
        """Initialize FailurePredictor with model artifact and decision threshold.

        Args:
            model_path: Path to the saved joblib Pipeline artifact.
            threshold_path: Path to the threshold analysis JSON report.
            fallback_threshold: Default threshold if report is unavailable (default: 0.45).

        Raises:
            FileNotFoundError: If model artifact does not exist.
            ValueError: If model artifact or threshold report is malformed.
            TypeError: If model artifact does not support predict_proba().
        """
        self.model_path = Path(model_path)
        self.threshold_path = Path(threshold_path) if threshold_path is not None else None
        self.fallback_threshold = float(fallback_threshold)

        self.pipeline: Pipeline = self._load_model(self.model_path)
        self.threshold: float = self._load_threshold(self.threshold_path, self.fallback_threshold)

    def _load_model(self, model_path: Path) -> Pipeline:
        """Load and validate the scikit-learn Pipeline artifact.

        Args:
            model_path: Path to model joblib file.

        Returns:
            Pipeline: Loaded and verified pipeline.

        Raises:
            FileNotFoundError: If artifact file is missing.
            ValueError: If artifact path is invalid or cannot be loaded.
            TypeError: If loaded object does not support predict_proba().
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at '{model_path}'.")
        if not model_path.is_file():
            raise ValueError(f"Model artifact path '{model_path}' is not a regular file.")

        try:
            pipeline = joblib.load(model_path)
        except Exception as exc:
            raise ValueError(f"Failed to load model artifact at '{model_path}': {exc}") from exc

        if not hasattr(pipeline, "predict_proba"):
            raise TypeError(
                f"Loaded artifact at '{model_path}' is not a valid classifier pipeline (missing predict_proba)."
            )

        return pipeline

    def _load_threshold(self, threshold_path: Optional[Path], fallback: float) -> float:
        """Load recommended classification threshold from analysis report or fallback.

        Args:
            threshold_path: Path to threshold JSON report.
            fallback: Documented fallback threshold (0.45).

        Returns:
            float: Applied decision threshold.

        Raises:
            ValueError: If report exists but is malformed JSON or has invalid threshold.
        """
        if threshold_path is None:
            logger.warning(
                "Threshold report path is None. Using documented fallback threshold: %.2f. Note: %s",
                fallback,
                MANDATORY_PROTOTYPE_DISCLAIMER,
            )
            return fallback

        if not threshold_path.exists():
            logger.warning(
                "Threshold report not found at '%s'. Using documented fallback threshold: %.2f. Note: %s",
                threshold_path,
                fallback,
                MANDATORY_PROTOTYPE_DISCLAIMER,
            )
            return fallback

        try:
            with open(threshold_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed threshold report at '{threshold_path}': Invalid JSON syntax ({exc})."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Error reading threshold report at '{threshold_path}': {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Malformed threshold report at '{threshold_path}': Root must be a JSON object."
            )

        rec = data.get("recommendation")
        if not isinstance(rec, dict) or "recommended_threshold" not in rec:
            raise ValueError(
                f"Malformed threshold report at '{threshold_path}': Missing 'recommendation.recommended_threshold' key."
            )

        thresh = rec["recommended_threshold"]
        if not isinstance(thresh, (int, float)) or not (0.0 <= thresh <= 1.0):
            raise ValueError(
                f"Malformed threshold report at '{threshold_path}': "
                f"Threshold must be a numeric value between 0.0 and 1.0, got {thresh!r}."
            )

        return float(thresh)

    def _validate_and_extract_features(
        self,
        telemetry: Union[MachineTelemetryInput, Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Validate input telemetry, enforce anti-leakage rules, and extract features.

        Guarantees:
        - Excludes UDI and Product ID.
        - Strictly forbids failure-mode diagnostics (TWF, HDF, PWF, OSF, RNF).
        - Validates all 6 approved features and their domain constraints.

        Args:
            telemetry: Raw input dict or MachineTelemetryInput model.

        Returns:
            Tuple[Dict[str, Any], Optional[str]]: (feature_dict_for_model, machine_id).

        Raises:
            ValueError: If input is invalid, contains forbidden columns, or fails validation.
        """
        if isinstance(telemetry, MachineTelemetryInput):
            return telemetry.to_feature_dict(), telemetry.machine_id

        if not isinstance(telemetry, dict):
            raise ValueError(
                f"Invalid telemetry input type: expected MachineTelemetryInput or dict, got {type(telemetry).__name__}."
            )

        # Anti-leakage verification: reject identifiers and failure modes
        forbidden_present = [
            k for k in telemetry.keys()
            if k in FORBIDDEN_COLUMNS or k.strip().lower() in [c.lower() for c in FORBIDDEN_COLUMNS]
        ]
        if forbidden_present:
            raise ValueError(
                f"Data leakage violation! Input contains forbidden column(s): {forbidden_present}. "
                f"FactoryPulse strictly prohibits passing identifiers or failure-mode diagnostics to inference."
            )

        # Normalize keys (map dataset column names to schema field names if needed)
        normalized: Dict[str, Any] = {}
        for key, val in telemetry.items():
            mapped_key = COLUMN_TO_FIELD_MAP.get(key, key)
            normalized[mapped_key] = val

        # Validate through Pydantic model contract
        try:
            validated = MachineTelemetryInput(**normalized)
        except ValidationError as exc:
            raise ValueError(f"Telemetry validation error: {exc}") from exc

        return validated.to_feature_dict(), validated.machine_id

    def predict(
        self,
        telemetry: Union[MachineTelemetryInput, Dict[str, Any]],
    ) -> PredictionResult:
        """Generate failure prediction for a single machine.

        Uses the saved sklearn Pipeline directly without manual scaling or encoding.
        Applies the explicit decision rule: probability >= threshold.

        Args:
            telemetry: Machine telemetry as MachineTelemetryInput or dict.

        Returns:
            PredictionResult: Structured assessment outcome.

        Raises:
            ValueError: If telemetry input fails validation or contains forbidden columns.
        """
        features, machine_id = self._validate_and_extract_features(telemetry)

        # Direct pipeline input: unscaled, original feature column names
        feature_df = pd.DataFrame([features], columns=APPROVED_FEATURE_NAMES)
        proba = float(self.pipeline.predict_proba(feature_df)[0, 1])

        # Explicit decision rule
        predicted_class = int(proba >= self.threshold)
        prediction_label = "Machine Failure" if predicted_class == 1 else "Normal Operation"

        return PredictionResult(
            failure_probability=proba,
            applied_threshold=self.threshold,
            predicted_class=predicted_class,
            prediction_label=prediction_label,
            machine_id=machine_id,
            model_name="RandomForestClassifier (balanced)",
            disclaimer=MANDATORY_PROTOTYPE_DISCLAIMER,
        )

    def predict_batch(
        self,
        telemetry_batch: Union[List[Union[MachineTelemetryInput, Dict[str, Any]]], BatchTelemetryInput],
    ) -> List[PredictionResult]:
        """Generate vectorized failure predictions for a batch of machines.

        Args:
            telemetry_batch: List of telemetry records or BatchTelemetryInput container.

        Returns:
            List[PredictionResult]: Structured assessment outcomes for all machines.

        Raises:
            ValueError: If batch is empty or any item fails validation.
        """
        if isinstance(telemetry_batch, BatchTelemetryInput):
            items = telemetry_batch.machines
        elif isinstance(telemetry_batch, list):
            items = telemetry_batch
        else:
            raise ValueError(
                f"Invalid batch input type: expected list or BatchTelemetryInput, got {type(telemetry_batch).__name__}."
            )

        if len(items) == 0:
            raise ValueError("Telemetry batch cannot be empty.")

        extracted = [self._validate_and_extract_features(item) for item in items]
        feature_dicts = [item[0] for item in extracted]
        machine_ids = [item[1] for item in extracted]

        # Vectorized pipeline inference
        feature_df = pd.DataFrame(feature_dicts, columns=APPROVED_FEATURE_NAMES)
        probas = self.pipeline.predict_proba(feature_df)[:, 1]

        results: List[PredictionResult] = []
        for proba, m_id in zip(probas, machine_ids):
            p = float(proba)
            pred_class = int(p >= self.threshold)
            label = "Machine Failure" if pred_class == 1 else "Normal Operation"
            results.append(
                PredictionResult(
                    failure_probability=p,
                    applied_threshold=self.threshold,
                    predicted_class=pred_class,
                    prediction_label=label,
                    machine_id=m_id,
                    model_name="RandomForestClassifier (balanced)",
                    disclaimer=MANDATORY_PROTOTYPE_DISCLAIMER,
                )
            )

        return results

    def get_service_info(self) -> Dict[str, Any]:
        """Retrieve operational metadata regarding the prediction service."""
        return {
            "model_path": str(self.model_path),
            "threshold_path": str(self.threshold_path) if self.threshold_path else None,
            "applied_threshold": self.threshold,
            "approved_features": APPROVED_FEATURE_NAMES,
            "model_name": "RandomForestClassifier (balanced)",
            "disclaimer": MANDATORY_PROTOTYPE_DISCLAIMER,
        }

