"""Baseline model configurations and scikit-learn pipeline constructors for FactoryPulse.

Combines ColumnTransformer preprocessing (OneHotEncoder for categorical 'Type'
and StandardScaler for continuous sensor metrics) with classification estimators.
Enclosing preprocessing and models within Pipelines strictly prevents data leakage.
"""

from typing import Dict, Final, List, Optional
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# Baseline Feature Definitions
# =============================================================================

CATEGORICAL_FEATURES: Final[List[str]] = ["Type"]

NUMERICAL_FEATURES: Final[List[str]] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

BASELINE_FEATURE_COLUMNS: Final[List[str]] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


# =============================================================================
# Preprocessing Constructor
# =============================================================================

def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that one-hot encodes 'Type' and scales numerical sensors.

    Uses handle_unknown='ignore' for OneHotEncoder so unexpected categories in live inference
    do not crash the pipeline.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    numerical_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            ("num", numerical_transformer, NUMERICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# =============================================================================
# Model Estimators & Combined Pipeline Constructors
# =============================================================================

def get_baseline_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Instantiate the 3 baseline classifiers specified for Phase 3.

    Args:
        random_state: Seed for reproducible estimators.

    Returns:
        Dict[str, BaseEstimator]: Dictionary mapping model identifiers to estimators.
    """
    return {
        "dummy": DummyClassifier(
            strategy="most_frequent",
        ),
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_estimators=100,
        ),
    }


def build_model_pipeline(
    model: BaseEstimator,
    preprocessor: Optional[BaseEstimator] = None,
) -> Pipeline:
    """Encapsulate preprocessing and classification into a single scikit-learn Pipeline.

    Enclosing both steps inside a Pipeline strictly prevents data leakage during training,
    cross-validation, and production inference.

    Args:
        model: BaseEstimator classifier.
        preprocessor: Optional preprocessor; defaults to build_preprocessor().

    Returns:
        Pipeline: Combined preprocessing + model pipeline.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
