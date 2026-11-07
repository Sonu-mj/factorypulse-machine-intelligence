"""Data loading and ingestion pipeline components for FactoryPulse.

Handles raw CSV ingestion safely with path verification, exception handling,
and schema validation execution.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import pandas as pd

from src.data.schema import (
    DataIngestionError,
    DataNotFoundError,
    DataValidationError,
)
from src.data.validator import ValidationReport, validate_dataset


def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame after validating file existence.

    Args:
        file_path: Path to the target CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        DataNotFoundError: If the file does not exist or is not a file.
        DataIngestionError: If the file is empty, malformed, or cannot be parsed.
    """
    path = Path(file_path)

    # 1. Verify file existence
    if not path.exists():
        raise DataNotFoundError(
            f"Dataset file not found at: '{path.resolve()}'. "
            "Please ensure the file exists and the path is correct."
        )

    if not path.is_file():
        raise DataNotFoundError(
            f"Specified path is not a regular file: '{path.resolve()}'. "
            "A path to a valid CSV file is required."
        )

    # 2. Load dataset safely with Pandas
    try:
        df = pd.read_csv(path)
        return df
    except pd.errors.EmptyDataError as err:
        raise DataIngestionError(
            f"Dataset file at '{path.resolve()}' is empty: {err}"
        ) from err
    except pd.errors.ParserError as err:
        raise DataIngestionError(
            f"Failed to parse CSV file at '{path.resolve()}': {err}"
        ) from err
    except Exception as err:
        raise DataIngestionError(
            f"Unexpected error loading dataset from '{path.resolve()}': {err}"
        ) from err


def load_and_validate_data(
    file_path: Union[str, Path] = "data/raw/ai4i2020.csv",
    raise_on_error: bool = False,
    expected_columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """Load a dataset from CSV and execute complete schema and data quality validation.

    Args:
        file_path: Path to dataset CSV. Defaults to 'data/raw/ai4i2020.csv'.
        raise_on_error: If True and validation fails, raises DataValidationError.
        expected_columns: Optional custom expected columns list.

    Returns:
        Tuple[pd.DataFrame, ValidationReport]: Loaded DataFrame and validation findings.

    Raises:
        DataNotFoundError: If the CSV file does not exist.
        DataIngestionError: If the CSV is empty or cannot be parsed.
        DataValidationError: If raise_on_error is True and validation criteria fail.
    """
    df = load_csv(file_path)
    report = validate_dataset(df, expected_columns=expected_columns)

    if raise_on_error and not report.is_valid:
        error_details = "\n - " + "\n - ".join(report.errors)
        raise DataValidationError(
            f"Dataset validation failed with {len(report.errors)} error(s):{error_details}"
        )

    return df, report
