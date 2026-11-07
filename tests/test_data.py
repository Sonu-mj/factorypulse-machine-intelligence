"""Unit and integration tests for FactoryPulse data ingestion, schema validation, and reporting.

Consolidates tests for CSV loading, error handling, schema checks, data validation rules,
and end-to-end ingestion pipeline execution (14 tests).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data.loader import load_and_validate_data, load_csv
from src.data.schema import (
    DataIngestionError,
    DataNotFoundError,
    DataValidationError,
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
)
from src.data.validator import validate_dataset

RAW_DATASET_PATH = Path("data/raw/ai4i2020.csv")


@pytest.fixture
def valid_sample_df():
    """Create a minimal valid DataFrame adhering to AI4I 2020 schema."""
    data = {
        "UDI": [1, 2, 3],
        "Product ID": ["M14860", "L47181", "L47182"],
        "Type": ["M", "L", "L"],
        "Air temperature [K]": [298.1, 298.2, 298.1],
        "Process temperature [K]": [308.6, 308.7, 308.5],
        "Rotational speed [rpm]": [1551, 1408, 1498],
        "Torque [Nm]": [42.8, 46.3, 49.4],
        "Tool wear [min]": [0, 3, 5],
        "Machine failure": [0, 0, 1],
        "TWF": [0, 0, 0],
        "HDF": [0, 0, 0],
        "PWF": [0, 0, 0],
        "OSF": [0, 0, 1],
        "RNF": [0, 0, 0],
    }
    return pd.DataFrame(data)


# =============================================================================
# 1. CSV Loader Tests
# =============================================================================

def test_load_csv_success():
    """Verify that existing AI4I 2020 dataset loads into a non-empty DataFrame."""
    df = load_csv(RAW_DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == (10000, 14)


def test_load_csv_file_not_found(tmp_path):
    """Verify that attempting to load a non-existent file raises DataNotFoundError."""
    non_existent = tmp_path / "missing_dataset.csv"
    with pytest.raises(DataNotFoundError) as exc_info:
        load_csv(non_existent)
    assert "Dataset file not found" in str(exc_info.value)
    assert str(non_existent.resolve()) in str(exc_info.value)


def test_load_csv_directory_path(tmp_path):
    """Verify that passing a directory instead of a file raises DataNotFoundError."""
    with pytest.raises(DataNotFoundError) as exc_info:
        load_csv(tmp_path)
    assert "not a regular file" in str(exc_info.value)


def test_load_csv_empty_file(tmp_path):
    """Verify that attempting to load an empty file raises DataIngestionError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()
    with pytest.raises(DataIngestionError) as exc_info:
        load_csv(empty_file)
    assert "is empty" in str(exc_info.value)


# =============================================================================
# 2. Ingestion Pipeline Orchestration Tests
# =============================================================================

def test_load_and_validate_raw_dataset():
    """Verify that the official AI4I 2020 dataset passes end-to-end ingestion and validation."""
    df, report = load_and_validate_data(RAW_DATASET_PATH)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10000, 14)
    assert report.is_valid is True
    assert report.total_rows == 10000
    assert report.total_columns == 14
    assert len(report.missing_columns) == 0
    assert report.total_missing_values == 0
    assert report.duplicate_rows_count == 0
    assert report.target_is_binary is True
    assert set(report.target_unique_values) == {0, 1}


def test_load_and_validate_raise_on_error(tmp_path):
    """Verify load_and_validate_data raises DataValidationError when raise_on_error is True on invalid data."""
    corrupt_csv = tmp_path / "corrupt.csv"
    corrupt_csv.write_text("UDI,Product ID\n1,M14860\n", encoding="utf-8")

    with pytest.raises(DataValidationError) as exc_info:
        load_and_validate_data(corrupt_csv, raise_on_error=True)

    assert "Dataset validation failed" in str(exc_info.value)


# =============================================================================
# 3. Data Quality Validator Tests
# =============================================================================

def test_validate_dataset_valid_sample(valid_sample_df):
    """Test validation on a fully valid sample DataFrame."""
    report = validate_dataset(valid_sample_df)
    assert report.is_valid is True
    assert report.total_rows == 3
    assert report.total_columns == 14
    assert len(report.missing_columns) == 0
    assert report.total_missing_values == 0
    assert report.duplicate_rows_count == 0
    assert report.target_is_binary is True
    assert report.target_unique_values == [0, 1]
    assert len(report.errors) == 0


def test_validate_dataset_missing_columns(valid_sample_df):
    """Test that missing required columns are detected and flagged as errors."""
    dropped_df = valid_sample_df.drop(columns=["Torque [Nm]", "UDI"])
    report = validate_dataset(dropped_df)
    assert report.is_valid is False
    assert "Torque [Nm]" in report.missing_columns
    assert "UDI" in report.missing_columns
    assert any("Missing required columns" in err for err in report.errors)


def test_validate_dataset_detects_missing_values(valid_sample_df):
    """Test that missing values are correctly counted per column and in total."""
    df_with_nans = valid_sample_df.copy()
    df_with_nans.loc[0, "Rotational speed [rpm]"] = None
    df_with_nans.loc[1, "Air temperature [K]"] = None
    report = validate_dataset(df_with_nans)
    assert report.total_missing_values == 2
    assert report.missing_values_by_column["Rotational speed [rpm]"] == 1
    assert report.missing_values_by_column["Air temperature [K]"] == 1
    assert any("missing value" in warn for warn in report.warnings)


def test_validate_dataset_detects_duplicates(valid_sample_df):
    """Test that duplicate rows are counted."""
    duplicate_row = valid_sample_df.iloc[[0]]
    df_with_dup = pd.concat([valid_sample_df, duplicate_row], ignore_index=True)
    report = validate_dataset(df_with_dup)
    assert report.duplicate_rows_count == 1
    assert any("duplicate row" in warn for warn in report.warnings)


def test_validate_dataset_target_non_binary(valid_sample_df):
    """Test that target column containing values other than 0 and 1 fails validation."""
    invalid_df = valid_sample_df.copy()
    invalid_df.loc[0, TARGET_COLUMN] = 2
    report = validate_dataset(invalid_df)
    assert report.is_valid is False
    assert report.target_is_binary is False
    assert any("must contain only binary values 0 and 1" in err for err in report.errors)


def test_validate_dataset_missing_target(valid_sample_df):
    """Test that a missing target column causes validation failure."""
    no_target_df = valid_sample_df.drop(columns=[TARGET_COLUMN])
    report = validate_dataset(no_target_df)
    assert report.is_valid is False
    assert report.target_is_binary is False
    assert any(f"Target column '{TARGET_COLUMN}' is missing" in err for err in report.errors)


def test_validate_dataset_empty_df():
    """Test that an empty DataFrame fails validation."""
    empty_df = pd.DataFrame(columns=EXPECTED_COLUMNS)
    report = validate_dataset(empty_df)
    assert report.is_valid is False
    assert any("contains 0 rows" in err for err in report.errors)


def test_validation_report_summary_and_dict(valid_sample_df):
    """Test report serialization to dict and human-readable summary string."""
    report = validate_dataset(valid_sample_df)
    report_dict = report.to_dict()
    assert isinstance(report_dict, dict)
    assert report_dict["is_valid"] is True
    assert report_dict["total_rows"] == 3

    summary = report.summary()
    assert "FACTORYPULSE DATA VALIDATION REPORT" in summary
    assert "Overall Status       : PASSED" in summary
    assert "Total Records (Rows) : 3" in summary

