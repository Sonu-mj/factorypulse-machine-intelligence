"""Baseline training pipeline orchestrator for FactoryPulse.

Coordinates data loading, feature validation, 80/20 stratified splitting, pipeline assembly,
model training, evaluation, and artifact saving to models/ and reports/.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_and_validate_data
from src.data.schema import FAILURE_MODE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN
from src.training.evaluate import (
    evaluate_pipeline,
    generate_evaluation_visualizations,
    save_model_comparison_report,
)
from src.training.models import (
    BASELINE_FEATURE_COLUMNS,
    build_model_pipeline,
    build_preprocessor,
    get_baseline_models,
)


def extract_baseline_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract strictly the 6 permitted baseline features and the target.

    Guarantees:
    - UDI and Product ID are excluded.
    - TWF, HDF, PWF, OSF, RNF are strictly excluded to prevent target leakage.
    - Exactly the 6 required features are returned in X.

    Args:
        df: Input DataFrame from ingestion pipeline.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X_features, y_target).

    Raises:
        ValueError: If any required feature or target is missing.
        RuntimeError: If any leakage column is detected in features.
    """
    # 1. Target presence check
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    y = df[TARGET_COLUMN].copy()

    # 2. Verify all 6 required baseline features exist
    missing_features = [col for col in BASELINE_FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required baseline feature(s): {missing_features}")

    # Extract exactly the 6 permitted features
    X = df[BASELINE_FEATURE_COLUMNS].copy()

    # 3. Anti-leakage verification: ensure no failure mode or identifier is in X
    forbidden_columns = set(IDENTIFIER_COLUMNS) | set(FAILURE_MODE_COLUMNS)
    for col in X.columns:
        if col in forbidden_columns:
            raise RuntimeError(
                f"Data leakage violation! '{col}' is forbidden but present in feature matrix."
            )

    return X, y


def run_baseline_training(
    dataset_path: Union[str, Path] = "data/raw/ai4i2020.csv",
    models_dir: Union[str, Path] = "models",
    reports_dir: Union[str, Path] = "reports",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete Phase 3 baseline model training, evaluation, and artifact generation.

    Args:
        dataset_path: Path to dataset CSV.
        models_dir: Output directory for trained model pipelines.
        reports_dir: Output directory for reports and figures.
        test_size: Proportion of dataset for test split (default 0.2).
        random_state: Seed for reproducible split and estimators (default 42).

    Returns:
        Dict[str, Any]: Complete results dictionary with model metrics and saved paths.
    """
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    figures_path = reports_path / "figures"

    models_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Load data using Phase 1 ingestion pipeline
    df, validation_report = load_and_validate_data(dataset_path, raise_on_error=True)

    # 2. Extract strictly the 6 features and target
    X, y = extract_baseline_data(df)

    # 3. 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 4. Instantiate preprocessor and baseline models
    raw_models = get_baseline_models(random_state=random_state)
    trained_pipelines = {}
    eval_results = {}
    saved_model_paths = {}

    for model_name, model_instance in raw_models.items():
        # Build individual pipeline containing a fresh preprocessor + classifier
        pipeline = build_model_pipeline(
            model=model_instance,
            preprocessor=build_preprocessor(),
        )

        # Fit pipeline on training split only (zero leakage)
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        # Evaluate on held-out test split
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        eval_results[model_name] = metrics

        # Save fitted pipeline under models/
        model_file = models_path / f"{model_name}_pipeline.joblib"
        joblib.dump(pipeline, model_file)
        saved_model_paths[model_name] = str(model_file)

    # 5. Generate and save visualizations under reports/figures/
    saved_plots = generate_evaluation_visualizations(
        results_by_model=eval_results,
        y_test=y_test,
        output_dir=figures_path,
    )

    # 6. Save comparison reports under reports/
    md_report, json_report = save_model_comparison_report(
        results_by_model=eval_results,
        output_dir=reports_path,
    )

    return {
        "status": "SUCCESS",
        "features_used": BASELINE_FEATURE_COLUMNS,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models_evaluated": list(raw_models.keys()),
        "evaluation_metrics": eval_results,
        "saved_models": saved_model_paths,
        "saved_plots": {k: str(v) for k, v in saved_plots.items()},
        "markdown_report": str(md_report),
        "json_report": str(json_report),
    }

