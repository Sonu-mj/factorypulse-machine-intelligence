"""Exploratory Data Analysis and Feature Engineering module for FactoryPulse."""

from src.features.eda import (
    EXCLUDED_FROM_PREDICTIVE_FEATURES,
    PREDICTIVE_NUMERICAL_FEATURES,
    analyze_feature_correlations,
    analyze_numerical_distributions,
    analyze_target_distribution,
    compare_operating_measurements,
    generate_dataset_overview,
    generate_eda_visualizations,
    run_eda_pipeline,
    save_eda_report,
)
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

__all__ = [
    # EDA
    "PREDICTIVE_NUMERICAL_FEATURES",
    "EXCLUDED_FROM_PREDICTIVE_FEATURES",
    "generate_dataset_overview",
    "analyze_target_distribution",
    "analyze_numerical_distributions",
    "analyze_feature_correlations",
    "compare_operating_measurements",
    "generate_eda_visualizations",
    "save_eda_report",
    "run_eda_pipeline",
    # Feature Engineering & Preprocessing
    "ENGINEERED_FEATURE_NAMES",
    "FINAL_FEATURE_NAMES",
    "engineer_features",
    "prepare_features_and_target",
    "split_dataset",
    "fit_and_apply_scaler",
    "run_feature_pipeline",
]
