"""Dataset schema definition, column constants, and custom data exceptions for FactoryPulse."""

from typing import Final, List

# =============================================================================
# Custom Ingestion & Validation Exceptions
# =============================================================================

class DataIngestionError(Exception):
    """Base exception for all data ingestion and processing errors."""
    pass


class DataNotFoundError(DataIngestionError):
    """Raised when the specified dataset file is not found."""
    pass


class DataValidationError(DataIngestionError):
    """Raised when dataset fails required validation criteria."""
    pass


# =============================================================================
# Column Constants
# =============================================================================

# Complete list of 14 expected columns in the AI4I 2020 dataset
EXPECTED_COLUMNS: Final[List[str]] = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# Primary identifier columns
IDENTIFIER_COLUMNS: Final[List[str]] = [
    "UDI",
    "Product ID",
]

# Machine failure target column (0 = no failure, 1 = failure)
TARGET_COLUMN: Final[str] = "Machine failure"

# Specific failure mode indicator columns.
# NOTE: These represent post-failure diagnostic modes and MUST NOT be used as
# predictive features for machine failure to prevent target data leakage.
FAILURE_MODE_COLUMNS: Final[List[str]] = [
    "TWF",  # Tool Wear Failure
    "HDF",  # Heat Dissipation Failure
    "PWF",  # Power Failure
    "OSF",  # Overstrain Failure
    "RNF",  # Random Failure
]

# Operational and physical sensor feature candidates
RAW_FEATURE_COLUMNS: Final[List[str]] = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
