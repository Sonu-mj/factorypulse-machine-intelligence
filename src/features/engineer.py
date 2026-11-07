"""Domain-driven feature engineering module for FactoryPulse.

Derives physically meaningful sensor features, encodes categorical machine types,
and strictly isolates predictor matrices from identifiers and target leakage columns.
"""

from typing import Final, List, Tuple
import numpy as np
import pandas as pd

from src.data.schema import FAILURE_MODE_COLUMNS, IDENTIFIER_COLUMNS, TARGET_COLUMN

# Physical & domain-engineered feature names
ENGINEERED_FEATURE_NAMES: Final[List[str]] = [
    "Temp difference [K]",
    "Mechanical power [W]",
    "Overstrain [min*Nm]",
    "Torque to speed ratio",
]

# Complete set of 12 clean predictive features
FINAL_FEATURE_NAMES: Final[List[str]] = [
    # Raw sensor signals (5)
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    # Physics-based engineered features (4)
    "Temp difference [K]",
    "Mechanical power [W]",
    "Overstrain [min*Nm]",
    "Torque to speed ratio",
    # One-hot encoded machine variants (3)
    "Type_H",
    "Type_L",
    "Type_M",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive industrial and physics-based features from raw machine telemetry.

    Features created:
    1. 'Temp difference [K]': Thermal gradient (Process temp - Air temp).
       Directly governs heat transfer and Heat Dissipation Failure (HDF).
    2. 'Mechanical power [W]': Rotational power = Torque * (Rotational speed * 2*pi/60).
       Power outside operational envelope (<3500 W or >9000 W) triggers Power Failure (PWF).
    3. 'Overstrain [min*Nm]': Interaction product of Tool wear and Torque.
       Excessive overstrain exceeds material tolerances (Overstrain Failure OSF).
    4. 'Torque to speed ratio': Torque / Rotational speed.
       Identifies high-torque/low-speed heavy cutting vs high-speed/low-torque idling.
    5. One-hot encoded 'Type': 'Type_H', 'Type_L', 'Type_M' as integer flags (0 or 1).

    Args:
        df: Input DataFrame containing raw columns.

    Returns:
        pd.DataFrame: A new DataFrame with all engineered features appended.
    """
    df_out = df.copy()

    # 1. Thermal dissipation gradient (Delta T)
    df_out["Temp difference [K]"] = (
        df_out["Process temperature [K]"] - df_out["Air temperature [K]"]
    )

    # 2. Rotational mechanical power (Watts)
    # Omega (rad/s) = rpm * (2 * pi / 60)
    angular_velocity = df_out["Rotational speed [rpm]"] * (2 * np.pi / 60)
    df_out["Mechanical power [W]"] = df_out["Torque [Nm]"] * angular_velocity

    # 3. Overstrain interaction product
    df_out["Overstrain [min*Nm]"] = (
        df_out["Tool wear [min]"] * df_out["Torque [Nm]"]
    )

    # 4. Torque to rotational speed ratio
    df_out["Torque to speed ratio"] = (
        df_out["Torque [Nm]"] / (df_out["Rotational speed [rpm]"].replace(0, np.nan))
    )

    # 5. One-hot encode machine Type ('L', 'M', 'H')
    if "Type" in df_out.columns:
        type_dummies = pd.get_dummies(df_out["Type"], prefix="Type", dtype=int)
        # Ensure all standard types exist even in test batches
        for standard_type in ["Type_H", "Type_L", "Type_M"]:
            if standard_type not in type_dummies.columns:
                type_dummies[standard_type] = 0
        df_out = pd.concat(
            [df_out, type_dummies[["Type_H", "Type_L", "Type_M"]]], axis=1
        )

    return df_out


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Engineer features, strictly drop leakage columns and identifiers, and return (X, y).

    Guarantees:
    - Identifiers ('UDI', 'Product ID') are dropped to prevent spurious memorization.
    - Failure mode columns ('TWF', 'HDF', 'PWF', 'OSF', 'RNF') are dropped to prevent target leakage.
    - Features are ordered deterministically matching FINAL_FEATURE_NAMES.

    Args:
        df: Validated dataset DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X_features, y_target).

    Raises:
        ValueError: If target column is missing.
        RuntimeError: If any data leakage column is detected in X.
    """
    df_engineered = engineer_features(df)

    # Extract target
    if TARGET_COLUMN not in df_engineered.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
    y = df_engineered[TARGET_COLUMN].copy()

    # Safety: ensure all final features exist
    available_features = [col for col in FINAL_FEATURE_NAMES if col in df_engineered.columns]
    X = df_engineered[available_features].copy()

    # Final safety check: ensure no leakage column slipped in
    for leakage_col in FAILURE_MODE_COLUMNS + IDENTIFIER_COLUMNS:
        if leakage_col in X.columns:
            raise RuntimeError(
                f"Data leakage detected! '{leakage_col}' is present in feature matrix."
            )

    return X, y

