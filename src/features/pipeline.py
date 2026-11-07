"""Feature engineering and preprocessing pipeline orchestrator for FactoryPulse.

Executes domain feature derivation, stratified dataset splitting, leakage-free
standardization, and persistence of processed datasets and transformer artifacts.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_and_validate_data
from src.data.schema import TARGET_COLUMN
from src.features.engineer import (
    FINAL_FEATURE_NAMES,
    prepare_features_and_target,
)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Execute stratified train/test split preserving severe class imbalance proportion.

    Args:
        X: Feature matrix.
        y: Target series (Machine failure).
        test_size: Proportion of dataset allocated to test set (default 0.2).
        random_state: Random seed for reproducibility (default 42).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def fit_and_apply_scaler(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler strictly on training set and transform both train and test.

    Strictly adheres to ML anti-leakage best practice: the scaler is fit ONLY
    on X_train and applied to X_test. Binary one-hot flags ('Type_*') are left
    unscaled as clean 0/1 indicators.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
            - X_train_scaled (DataFrame with identical columns and index)
            - X_test_scaled (DataFrame with identical columns and index)
            - Fitted StandardScaler instance
    """
    # Identify continuous numerical columns to scale
    numerical_cols = [c for c in X_train.columns if not c.startswith("Type_")]
    categorical_cols = [c for c in X_train.columns if c.startswith("Type_")]

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_num_scaled = scaler.fit_transform(X_train[numerical_cols])
    X_test_num_scaled = scaler.transform(X_test[numerical_cols])

    # Reconstruct DataFrame with original index and columns
    df_train_num = pd.DataFrame(
        X_train_num_scaled,
        columns=numerical_cols,
        index=X_train.index,
    )
    df_test_num = pd.DataFrame(
        X_test_num_scaled,
        columns=numerical_cols,
        index=X_test.index,
    )

    # Combine scaled numerical features with unscaled one-hot categorical features
    X_train_scaled = pd.concat([df_train_num, X_train[categorical_cols]], axis=1)[
        X_train.columns
    ]
    X_test_scaled = pd.concat([df_test_num, X_test[categorical_cols]], axis=1)[
        X_test.columns
    ]

    return X_train_scaled, X_test_scaled, scaler


def run_feature_pipeline(
    raw_csv_path: Union[str, Path] = "data/raw/ai4i2020.csv",
    output_dir: Union[str, Path] = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete end-to-end feature engineering and preprocessing workflow.

    Workflow:
    1. Loads and validates raw data using Phase 1 pipeline.
    2. Derives domain physics features and isolates feature matrix X and target y.
    3. Performs stratified train/test split.
    4. Fits StandardScaler on training set only and transforms both splits.
    5. Saves processed train/test datasets (unscaled and scaled) and scaler artifact to data/processed/.

    Args:
        raw_csv_path: Path to raw dataset CSV.
        output_dir: Destination directory for processed datasets.
        test_size: Proportion of dataset for test split (default 0.2).
        random_state: Seed for reproducible split (default 42).

    Returns:
        Dict[str, Any]: Metadata detailing processed shapes, class distributions, and paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and validate
    df_raw, validation_report = load_and_validate_data(raw_csv_path, raise_on_error=True)

    # 2. Derive features and separate target
    X, y = prepare_features_and_target(df_raw)

    # 3. Stratified train/test split
    X_train, X_test, y_train, y_test = split_dataset(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Standardize numerical features (fit strictly on X_train)
    X_train_scaled, X_test_scaled, scaler = fit_and_apply_scaler(X_train, X_test)

    # 5. Persist processed datasets and transformer artifact
    train_unscaled_path = out_dir / "train.csv"
    test_unscaled_path = out_dir / "test.csv"
    train_scaled_path = out_dir / "train_scaled.csv"
    test_scaled_path = out_dir / "test_scaled.csv"
    scaler_artifact_path = out_dir / "scaler.joblib"

    # Save unscaled datasets (features + target)
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv(train_unscaled_path, index=False)
    test_df.to_csv(test_unscaled_path, index=False)

    # Save scaled datasets (features + target)
    train_scaled_df = pd.concat([X_train_scaled, y_train], axis=1)
    test_scaled_df = pd.concat([X_test_scaled, y_test], axis=1)
    train_scaled_df.to_csv(train_scaled_path, index=False)
    test_scaled_df.to_csv(test_scaled_path, index=False)

    # Save fitted scaler for inference
    joblib.dump(scaler, scaler_artifact_path)

    # Calculate class distribution statistics
    train_failure_rate = float(y_train.mean())
    test_failure_rate = float(y_test.mean())

    return {
        "status": "SUCCESS",
        "validation_passed": validation_report.is_valid,
        "features_count": len(FINAL_FEATURE_NAMES),
        "feature_names": FINAL_FEATURE_NAMES,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_failure_rate": train_failure_rate,
        "test_failure_rate": test_failure_rate,
        "paths": {
            "train_unscaled": str(train_unscaled_path),
            "test_unscaled": str(test_unscaled_path),
            "train_scaled": str(train_scaled_path),
            "test_scaled": str(test_scaled_path),
            "scaler_artifact": str(scaler_artifact_path),
        },
    }

