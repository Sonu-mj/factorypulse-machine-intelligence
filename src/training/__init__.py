"""Model training and evaluation module for FactoryPulse."""

from src.training.evaluate import (
    evaluate_pipeline,
    generate_evaluation_visualizations,
    save_model_comparison_report,
)
from src.training.models import (
    BASELINE_FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
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

__all__ = [
    "BASELINE_FEATURE_COLUMNS",
    "CATEGORICAL_FEATURES",
    "NUMERICAL_FEATURES",
    "build_preprocessor",
    "get_baseline_models",
    "build_model_pipeline",
    "evaluate_pipeline",
    "generate_evaluation_visualizations",
    "save_model_comparison_report",
    "extract_baseline_data",
    "run_baseline_training",
    "DEFAULT_CANDIDATE_THRESHOLDS",
    "evaluate_threshold_candidates_cv",
    "select_recommended_threshold",
    "evaluate_threshold_on_test",
    "run_threshold_analysis",
]
