"""Unit and integration tests for Phase 3 Feature Engineering and Preprocessing Pipeline."""

from pathlib import Path
import numpy as np
import pytest
import pandas as pd

from src.data.loader import load_and_validate_data
from src.data.schema import FAILURE_MODE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN
from src.features.engineer import (
    ENGINEERED_FEATURE_NAMES,
    FINAL_FEATURE_NAMES,
    engineer_features,
    prepare_features_and_target,
)
from src.features.pipeline import (
    fit_and_apply_scaler,
    run_feature_pipeline,
    split_dataset,
)


@pytest.fixture(scope="module")
def raw_df():
    """Load and validate raw AI4I 2020 dataset."""
    df, report = load_and_validate_data("data/raw/ai4i2020.csv")
    assert report.is_valid is True
    return df


def test_engineer_features_formula_accuracy(raw_df):
    """Verify that domain physics formulas for features are calculated correctly."""
    df_eng = engineer_features(raw_df)

    # 1. Verify engineered feature columns exist
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in df_eng.columns

    # 2. Temperature difference: Delta T = Process temp - Air temp
    expected_temp_diff = (
        raw_df["Process temperature [K]"] - raw_df["Air temperature [K]"]
    )
    pd.testing.assert_series_equal(
        df_eng["Temp difference [K]"], expected_temp_diff, check_names=False
    )

    # 3. Mechanical power: P = Torque * (rpm * 2*pi / 60)
    expected_power = (
        raw_df["Torque [Nm]"] * raw_df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    )
    pd.testing.assert_series_equal(
        df_eng["Mechanical power [W]"], expected_power, check_names=False
    )

    # 4. Overstrain product: Wear * Torque
    expected_overstrain = raw_df["Tool wear [min]"] * raw_df["Torque [Nm]"]
    pd.testing.assert_series_equal(
        df_eng["Overstrain [min*Nm]"], expected_overstrain, check_names=False
    )

    # 5. One-hot encoded Type
    assert set(["Type_H", "Type_L", "Type_M"]).issubset(set(df_eng.columns))
    # Sum of one-hot flags per row must equal 1
    type_sum = df_eng["Type_H"] + df_eng["Type_L"] + df_eng["Type_M"]
    assert (type_sum == 1).all()


def test_prepare_features_and_target_leakage_prevention(raw_df):
    """Verify feature matrix X excludes all leakage failure modes and identifiers."""
    X, y = prepare_features_and_target(raw_df)

    # 1. Feature dimensions
    assert X.shape == (10000, 12)
    assert list(X.columns) == FINAL_FEATURE_NAMES
    assert len(y) == 10000
    assert y.name == TARGET_COLUMN

    # 2. Strict leakage assertion: No failure mode in X
    for leakage_col in FAILURE_MODE_COLUMNS:
        assert leakage_col not in X.columns

    # 3. No identifiers in X
    for id_col in IDENTIFIER_COLUMNS:
        assert id_col not in X.columns

    # 4. No raw string Type column in X (replaced by Type_H, Type_L, Type_M)
    assert "Type" not in X.columns


def test_split_dataset_stratified(raw_df):
    """Verify stratified splitting maintains identical minority class proportion across splits."""
    X, y = prepare_features_and_target(raw_df)
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=0.2, random_state=42
    )

    # Size verification
    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert len(y_train) == 8000
    assert len(y_test) == 2000

    # Failure rate preservation (stratified)
    orig_rate = y.mean()
    train_rate = y_train.mean()
    test_rate = y_test.mean()

    assert train_rate == pytest.approx(orig_rate, abs=0.001)
    assert test_rate == pytest.approx(orig_rate, abs=0.002)


def test_fit_and_apply_scaler_leakage_free(raw_df):
    """Verify scaler fits strictly on X_train (mean ~0, std ~1) and leaves one-hot flags unscaled."""
    X, y = prepare_features_and_target(raw_df)
    X_train, X_test, _, _ = split_dataset(X, y, test_size=0.2, random_state=42)

    X_train_scaled, X_test_scaled, scaler = fit_and_apply_scaler(X_train, X_test)

    # Check numerical columns in train are standardized
    numerical_cols = [c for c in X_train.columns if not c.startswith("Type_")]
    for col in numerical_cols:
        assert X_train_scaled[col].mean() == pytest.approx(0.0, abs=1e-6)
        assert X_train_scaled[col].std(ddof=0) == pytest.approx(1.0, abs=1e-6)

    # Check categorical one-hot columns remain strictly 0 and 1
    categorical_cols = ["Type_H", "Type_L", "Type_M"]
    for col in categorical_cols:
        assert set(X_train_scaled[col].unique()).issubset({0, 1})
        assert set(X_test_scaled[col].unique()).issubset({0, 1})


def test_run_feature_pipeline_end_to_end(tmp_path):
    """Verify end-to-end feature pipeline execution and artifact persistence."""
    results = run_feature_pipeline(
        raw_csv_path="data/raw/ai4i2020.csv",
        output_dir=tmp_path,
        test_size=0.2,
        random_state=42,
    )

    assert results["status"] == "SUCCESS"
    assert results["validation_passed"] is True
    assert results["features_count"] == 12
    assert results["train_samples"] == 8000
    assert results["test_samples"] == 2000

    # Verify generated disk files
    train_unscaled = tmp_path / "train.csv"
    test_unscaled = tmp_path / "test.csv"
    train_scaled = tmp_path / "train_scaled.csv"
    test_scaled = tmp_path / "test_scaled.csv"
    scaler_file = tmp_path / "scaler.joblib"

    assert train_unscaled.exists()
    assert test_unscaled.exists()
    assert train_scaled.exists()
    assert test_scaled.exists()
    assert scaler_file.exists()

    # Verify row counts in saved CSVs
    df_train = pd.read_csv(train_scaled)
    df_test = pd.read_csv(test_scaled)
    assert df_train.shape == (8000, 13)  # 12 features + 1 target
    assert df_test.shape == (2000, 13)
    assert TARGET_COLUMN in df_train.columns
    assert TARGET_COLUMN in df_test.columns

