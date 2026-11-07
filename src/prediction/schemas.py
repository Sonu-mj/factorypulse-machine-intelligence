"""Pydantic schemas and data contracts for FactoryPulse prediction engine.

Defines validated input/output data schemas for single and batch inference.
Strictly excludes identifier columns (UDI, Product ID) and failure-mode diagnostics
(TWF, HDF, PWF, OSF, RNF) to prevent data leakage.
"""

from typing import Any, Dict, Final, List, Literal, Optional
from pydantic import BaseModel, Field

# Mandatory disclaimer required across all prototype predictions
MANDATORY_PROTOTYPE_DISCLAIMER: Final[str] = (
    "EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT"
)

# Canonical 6 approved model features
APPROVED_FEATURE_NAMES: Final[List[str]] = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Forbidden columns that must never be passed to model inference
FORBIDDEN_COLUMNS: Final[List[str]] = [
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Machine failure",
]


class MachineTelemetryInput(BaseModel):
    """Telemetry data contract for a single industrial machine.

    Accepts strictly the 6 approved physical sensor and variant attributes,
    with an optional tracking machine_id metadata field.
    """

    machine_id: Optional[str] = Field(
        default=None,
        description="Optional machine identifier for operational tracking (e.g., CNC-01).",
        examples=["CNC-01"],
    )
    type: Literal["L", "M", "H"] = Field(
        ...,
        description="Machine product variant: 'L' (Low), 'M' (Medium), or 'H' (High).",
        examples=["M"],
    )
    air_temperature_k: float = Field(
        ...,
        gt=0.0,
        description="Ambient air temperature in Kelvin [K]. Must be positive.",
        examples=[298.5],
    )
    process_temperature_k: float = Field(
        ...,
        gt=0.0,
        description="Process operating temperature in Kelvin [K]. Must be positive.",
        examples=[308.7],
    )
    rotational_speed_rpm: int = Field(
        ...,
        gt=0,
        description="Spindle rotational speed in revolutions per minute [rpm]. Must be positive.",
        examples=[1450],
    )
    torque_nm: float = Field(
        ...,
        ge=0.0,
        description="Spindle torque in Newton-meters [Nm]. Must be non-negative.",
        examples=[45.0],
    )
    tool_wear_min: int = Field(
        ...,
        ge=0,
        description="Cumulative tool wear in minutes [min]. Must be non-negative.",
        examples=[120],
    )

    def to_feature_dict(self) -> Dict[str, Any]:
        """Map snake_case schema fields into the exact column names required by the model pipeline."""
        return {
            "Type": self.type,
            "Air temperature [K]": float(self.air_temperature_k),
            "Process temperature [K]": float(self.process_temperature_k),
            "Rotational speed [rpm]": int(self.rotational_speed_rpm),
            "Torque [Nm]": float(self.torque_nm),
            "Tool wear [min]": int(self.tool_wear_min),
        }


class BatchTelemetryInput(BaseModel):
    """Container schema for batch telemetry scoring."""

    machines: List[MachineTelemetryInput] = Field(
        ...,
        min_length=1,
        description="Non-empty list of machine telemetry records to evaluate.",
    )


class PredictionResult(BaseModel):
    """Structured failure risk assessment for an individual machine.

    Conforms to Phase 4 requirements:
    - Failure probability
    - Applied threshold
    - Predicted class (0 or 1)
    - Prediction label ('Normal Operation' or 'Machine Failure')
    - Optional machine_id metadata

    Note on Risk Tiers: Arbitrary risk-tier boundaries (e.g. LOW/MEDIUM/HIGH/CRITICAL)
    are intentionally omitted because the project has no approved domain-validated
    risk-tier definitions.
    """

    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of machine failure (0.0 to 1.0).",
        examples=[0.485],
    )
    applied_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classification decision threshold applied (e.g., 0.45).",
        examples=[0.45],
    )
    predicted_class: int = Field(
        ...,
        description="Binary classification outcome: 0 = Normal Operation, 1 = Machine Failure.",
        examples=[1],
    )
    prediction_label: str = Field(
        ...,
        description="Human-readable classification outcome: 'Normal Operation' or 'Machine Failure'.",
        examples=["Machine Failure"],
    )
    machine_id: Optional[str] = Field(
        default=None,
        description="Machine identifier if provided in the input.",
        examples=["CNC-01"],
    )
    model_name: str = Field(
        default="RandomForestClassifier (balanced)",
        description="Name of the underlying trained model pipeline.",
    )
    disclaimer: str = Field(
        default=MANDATORY_PROTOTYPE_DISCLAIMER,
        description="Mandatory prototype status disclaimer.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return self.model_dump()


class BatchPredictionResult(BaseModel):
    """Aggregated batch prediction response."""

    total_evaluated: int = Field(
        ...,
        description="Total number of machines evaluated in the batch.",
    )
    predicted_failures_count: int = Field(
        ...,
        description="Total number of machines predicted to fail (class 1).",
    )
    applied_threshold: float = Field(
        ...,
        description="Classification decision threshold applied across the batch.",
    )
    predictions: List[PredictionResult] = Field(
        ...,
        description="List of structured prediction results for each machine.",
    )
    disclaimer: str = Field(
        default=MANDATORY_PROTOTYPE_DISCLAIMER,
        description="Mandatory prototype status disclaimer.",
    )


# Backward compatibility aliases
PredictionOutput = PredictionResult
BatchPredictionOutput = BatchPredictionResult
