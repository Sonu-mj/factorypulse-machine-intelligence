"""Unit and integration tests for Phase 3 Baseline Model Development and Phase 3.5 Threshold Analysis.

Consolidates tests for feature extraction, anti-leakage guards, preprocessing transformations,
model pipeline fitting, baseline evaluation metrics, out-of-fold CV threshold optimization,
and test-set evaluation deltas (10 tests).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.loader import load_and_validate_data
from src.data.schema import FAILURE_MODE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN
from src.training.evaluate import evaluate_pipeline
from src.training.models import (
    BASELINE_FEATURE_COLUMNS,
    build_model_pipeline,
    build_preprocessor,
    get_baseline_models,
)
from src.training.pipeline import extract_baseline_data, run_baseline_training
from src.training.threshold_analysis import (
    DEFAULT_CANDIDATE_THRESHOLDS,
    evaluate_threshold_candidates_cv,
    evaluate_threshold_on_test,
    run_threshold_analysis,
    select_recommended_threshold,
)


@pytest.fixture(scope="module")
def raw_df():
    """Load and validate raw AI4I 2020 dataset."""
    df, report = load_and_validate_data("data/raw/ai4i2020.csv")
    assert report.is_valid is True
    return df


@pytest.fixture(scope="module")
def train_test_data(raw_df):
    """Return stratified 80/20 train/test splits from raw data."""
    X, y = extract_baseline_data(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# 1. Feature Extraction & Anti-Leakage Tests
# =============================================================================

def test_extract_baseline_data_integrity(raw_df):
    """Verify extract_baseline_data returns strictly the 6 features and target."""
    X, y = extract_baseline_data(raw_df)

    assert X.shape == (10000, 6)
    assert list(X.columns) == BASELINE_FEATURE_COLUMNS
    assert len(y) == 10000
    assert y.name == TARGET_COLUMN
    assert set(y.unique()) == {0, 1}


def test_extract_baseline_data_leakage_exclusion(raw_df):
    """Verify that none of the failure mode columns or identifiers are present in X."""
    X, _ = extract_baseline_data(raw_df)

    for leakage_col in FAILURE_MODE_COLUMNS:
        assert leakage_col not in X.columns

    for id_col in IDENTIFIER_COLUMNS:
        assert id_col not in X.columns


# =============================================================================
# 2. Preprocessor & Model Pipeline Creation Tests
# =============================================================================

def test_build_preprocessor_transformation(raw_df):
    """Verify ColumnTransformer transforms 1 categorical + 5 numerical features into 8 output columns."""
    X, _ = extract_baseline_data(raw_df)
    preprocessor = build_preprocessor()

    X_trans = preprocessor.fit_transform(X)
    assert X_trans.shape == (10000, 8)
    assert not np.isnan(X_trans).any()


def test_model_pipelines_creation_and_fitting(raw_df):
    """Verify all 3 baseline pipelines can be built, fitted, and produce valid predictions."""
    X, y = extract_baseline_data(raw_df)
    X_train, X_test = X.iloc[:8000], X.iloc[8000:]
    y_train, y_test = y.iloc[:8000], y.iloc[8000:]

    models = get_baseline_models(random_state=42)
    assert set(models.keys()) == {"dummy", "logistic_regression", "random_forest"}

    for name, model in models.items():
        pipe = build_model_pipeline(model=model)
        assert isinstance(pipe, Pipeline)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        assert len(y_pred) == len(X_test)
        assert set(np.unique(y_pred)).issubset({0, 1})


def test_evaluate_pipeline_metrics_contract(raw_df):
    """Verify evaluate_pipeline returns all required metrics and confusion matrix values."""
    X, y = extract_baseline_data(raw_df)
    X_train, X_test = X.iloc[:8000], X.iloc[8000:]
    y_train, y_test = y.iloc[:8000], y.iloc[8000:]

    models = get_baseline_models(random_state=42)
    pipe = build_model_pipeline(model=models["logistic_regression"])
    pipe.fit(X_train, y_train)

    metrics = evaluate_pipeline(pipe, X_test, y_test)

    required_keys = ["confusion_matrix", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    for key in required_keys:
        assert key in metrics

    cm = metrics["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_test)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_run_baseline_training_end_to_end(tmp_path):
    """Verify end-to-end execution, artifact generation, and model re-loadability."""
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    results = run_baseline_training(
        dataset_path="data/raw/ai4i2020.csv",
        models_dir=models_dir,
        reports_dir=reports_dir,
        test_size=0.2,
        random_state=42,
    )

    assert results["status"] == "SUCCESS"
    assert results["train_samples"] == 8000
    assert results["test_samples"] == 2000

    for model_name in ["dummy", "logistic_regression", "random_forest"]:
        model_file = models_dir / f"{model_name}_pipeline.joblib"
        assert model_file.exists()
        reloaded_pipe = joblib.load(model_file)
        assert isinstance(reloaded_pipe, Pipeline)

        sample_input = pd.DataFrame([{
            "Type": "M",
            "Air temperature [K]": 298.5,
            "Process temperature [K]": 308.7,
            "Rotational speed [rpm]": 1450,
            "Torque [Nm]": 45.0,
            "Tool wear [min]": 50,
        }])
        pred = reloaded_pipe.predict(sample_input)
        assert pred[0] in [0, 1]

    md_report = reports_dir / "baseline_model_comparison.md"
    json_report = reports_dir / "baseline_model_comparison.json"
    assert md_report.exists()
    assert json_report.exists()

    figures_dir = reports_dir / "figures"
    assert (figures_dir / "confusion_matrices_comparison.png").exists()
    assert (figures_dir / "model_roc_pr_curves.png").exists()
    assert (figures_dir / "confusion_matrix_random_forest.png").exists()


# =============================================================================
# 3. Threshold Analysis & Optimization Tests
# =============================================================================

def test_evaluate_threshold_candidates_cv_properties(train_test_data):
    """Verify that out-of-fold CV produces valid metrics for all candidate thresholds."""
    X_train, _, y_train, _ = train_test_data
    rf_model = get_baseline_models(random_state=42)["random_forest"]
    pipeline = build_model_pipeline(model=rf_model, preprocessor=build_preprocessor())

    test_candidates = [0.20, 0.40, 0.60]
    cv_results, oof_proba = evaluate_threshold_candidates_cv(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        candidate_thresholds=test_candidates,
        n_splits=3,
        random_state=42,
    )

    assert len(cv_results) == 3
    assert len(oof_proba) == len(X_train)

    for res in cv_results:
        assert "threshold" in res
        assert "precision" in res
        assert "recall" in res
        assert "f1_score" in res
        assert res["tp"] + res["fp"] + res["fn"] + res["tn"] == len(X_train)

    recalls = [r["recall"] for r in cv_results]
    assert recalls[0] >= recalls[1] >= recalls[2]


def test_select_recommended_threshold_logic():
    """Verify selection of F1-optimal threshold from candidate results."""
    mock_results = [
        {"threshold": 0.30, "precision": 0.50, "recall": 0.80, "f1_score": 0.615},
        {"threshold": 0.45, "precision": 0.66, "recall": 0.68, "f1_score": 0.670},
        {"threshold": 0.60, "precision": 0.80, "recall": 0.50, "f1_score": 0.615},
    ]
    rec = select_recommended_threshold(mock_results)

    assert rec["recommended_threshold"] == 0.45
    assert rec["f1_optimal"]["threshold"] == 0.45
    assert rec["f1_optimal"]["f1_score"] == 0.670


def test_evaluate_threshold_on_test(train_test_data):
    """Verify single-pass test set evaluation computes accurate metrics and deltas."""
    X_train, X_test, y_train, y_test = train_test_data
    rf_model = get_baseline_models(random_state=42)["random_forest"]
    pipeline = build_model_pipeline(model=rf_model, preprocessor=build_preprocessor())
    pipeline.fit(X_train, y_train)

    eval_out = evaluate_threshold_on_test(
        fitted_pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        threshold=0.45,
    )

    assert eval_out["selected_threshold"] == 0.45
    assert "test_metrics_at_selected_threshold" in eval_out
    assert "test_metrics_at_default_0_50" in eval_out
    assert "delta_vs_default" in eval_out

    sel_m = eval_out["test_metrics_at_selected_threshold"]
    def_m = eval_out["test_metrics_at_default_0_50"]
    assert sel_m["tp"] + sel_m["fp"] + sel_m["fn"] + sel_m["tn"] == len(X_test)
    assert def_m["tp"] + def_m["fp"] + def_m["fn"] + def_m["tn"] == len(X_test)


def test_run_threshold_analysis_end_to_end_and_integrity(tmp_path):
    """Verify end-to-end execution, artifact generation, and integrity of reports."""
    reports_dir = tmp_path / "reports"

    results = run_threshold_analysis(
        dataset_path="data/raw/ai4i2020.csv",
        reports_dir=reports_dir,
        random_state=42,
    )

    assert results["status"] == "SUCCESS"
    assert len(results["cv_results"]) == len(DEFAULT_CANDIDATE_THRESHOLDS)
    assert results["recommendation"]["recommended_threshold"] > 0.0

    md_file = reports_dir / "rf_threshold_analysis.md"
    json_file = reports_dir / "rf_threshold_analysis.json"
    curve_plot = reports_dir / "figures" / "rf_threshold_vs_metrics.png"
    cm_plot = reports_dir / "figures" / "rf_confusion_matrix_threshold_comparison.png"

    assert md_file.exists()
    assert json_file.exists()
    assert curve_plot.exists()
    assert cm_plot.exists()
