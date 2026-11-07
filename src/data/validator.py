"""Data validation component for FactoryPulse.

Verifies schema integrity, missing values, duplicates, and target distribution,
producing a comprehensive, human-readable validation report.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from src.data.schema import EXPECTED_COLUMNS, TARGET_COLUMN


@dataclass
class ValidationReport:
    """Structured report detailing the findings of dataset validation."""

    is_valid: bool
    total_rows: int
    total_columns: int
    present_columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    unexpected_columns: List[str] = field(default_factory=list)
    missing_values_by_column: Dict[str, int] = field(default_factory=dict)
    total_missing_values: int = 0
    duplicate_rows_count: int = 0
    target_is_binary: bool = True
    target_unique_values: List[Any] = field(default_factory=list)
    target_distribution: Dict[Any, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation report to a standard Python dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "present_columns": self.present_columns,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "missing_values_by_column": self.missing_values_by_column,
            "total_missing_values": self.total_missing_values,
            "duplicate_rows_count": self.duplicate_rows_count,
            "target_is_binary": self.target_is_binary,
            "target_unique_values": self.target_unique_values,
            "target_distribution": self.target_distribution,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def summary(self) -> str:
        """Generate a clean, readable text summary of the validation report."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            "==================================================",
            f"       FACTORYPULSE DATA VALIDATION REPORT        ",
            "==================================================",
            f"Overall Status       : {status}",
            f"Total Records (Rows) : {self.total_rows}",
            f"Total Columns        : {self.total_columns}",
            "--------------------------------------------------",
            "1. Schema & Column Check:",
            f"   - Expected Columns ({len(EXPECTED_COLUMNS)}): {'All Present' if not self.missing_columns else 'MISSING'}",
        ]
        if self.missing_columns:
            lines.append(f"     * Missing: {', '.join(self.missing_columns)}")
        if self.unexpected_columns:
            lines.append(f"     * Additional unexpected: {', '.join(self.unexpected_columns)}")

        lines.extend([
            "--------------------------------------------------",
            "2. Missing Values Check:",
            f"   - Total Missing Cells: {self.total_missing_values}",
        ])
        if self.total_missing_values > 0:
            for col, count in self.missing_values_by_column.items():
                if count > 0:
                    lines.append(f"     * {col}: {count} missing ({count / self.total_rows:.2%})")
        else:
            lines.append("   - No missing values detected across all columns.")

        lines.extend([
            "--------------------------------------------------",
            "3. Duplicate Rows Check:",
            f"   - Duplicate Rows: {self.duplicate_rows_count}",
            "--------------------------------------------------",
            "4. Target Column Check ('Machine failure'):",
            f"   - Target Present    : {'Yes' if TARGET_COLUMN in self.present_columns else 'No'}",
            f"   - Strictly Binary   : {'Yes (0 and 1 only)' if self.target_is_binary else 'NO (Invalid values detected)'}",
            f"   - Unique Values     : {self.target_unique_values}",
            f"   - Class Breakdown   : {self.target_distribution}",
        ])

        if self.errors:
            lines.extend([
                "--------------------------------------------------",
                "CRITICAL ERRORS:",
                *[f"   [ERROR] {err}" for err in self.errors],
            ])

        if self.warnings:
            lines.extend([
                "--------------------------------------------------",
                "WARNINGS:",
                *[f"   [WARN] {warn}" for warn in self.warnings],
            ])

        lines.append("==================================================")
        return "\n".join(lines)


def validate_dataset(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
) -> ValidationReport:
    """Validate a loaded pandas DataFrame against FactoryPulse schema and quality criteria.

    Verification steps:
    1. Dataset is not empty.
    2. All expected columns (default 14) are present.
    3. Missing values count per column and overall.
    4. Exact duplicate rows count.
    5. 'Machine failure' contains only valid binary values {0, 1}.

    Args:
        df: Pandas DataFrame to validate.
        expected_columns: Optional list of required column names.
                          Defaults to EXPECTED_COLUMNS from schema.

    Returns:
        ValidationReport: Detailed validation findings and pass/fail status.
    """
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS

    errors: List[str] = []
    warnings: List[str] = []

    total_rows, total_columns = df.shape
    present_cols = list(df.columns)

    # 1. Non-empty DataFrame check
    if total_rows == 0:
        errors.append("Dataset contains 0 rows; data is empty.")

    # 2. Check for missing and unexpected columns
    expected_set = set(expected_columns)
    present_set = set(present_cols)

    missing_cols = [col for col in expected_columns if col not in present_set]
    unexpected_cols = [col for col in present_cols if col not in expected_set]

    if missing_cols:
        errors.append(
            f"Missing required columns ({len(missing_cols)}): {missing_cols}"
        )
    if unexpected_cols:
        warnings.append(
            f"Dataset has unexpected additional columns ({len(unexpected_cols)}): {unexpected_cols}"
        )

    # 3. Missing values check
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = sum(missing_by_col.values())
    if total_missing > 0:
        warnings.append(f"Dataset contains {total_missing} missing value(s).")

    # 4. Duplicate rows check
    duplicate_rows_count = int(df.duplicated().sum())
    if duplicate_rows_count > 0:
        warnings.append(f"Dataset contains {duplicate_rows_count} duplicate row(s).")

    # 5. Target column validation ('Machine failure')
    target_is_binary = True
    target_unique_values: List[Any] = []
    target_distribution: Dict[Any, int] = {}

    if TARGET_COLUMN not in df.columns:
        target_is_binary = False
        errors.append(f"Target column '{TARGET_COLUMN}' is missing from the dataset.")
    else:
        target_series = df[TARGET_COLUMN]
        # Check nulls in target
        target_nulls = target_series.isnull().sum()
        if target_nulls > 0:
            target_is_binary = False
            errors.append(
                f"Target column '{TARGET_COLUMN}' has {target_nulls} null/missing value(s)."
            )

        unique_vals = target_series.dropna().unique().tolist()
        # Sort for deterministic reporting if comparable
        try:
            target_unique_values = sorted(unique_vals)
        except Exception:
            target_unique_values = unique_vals

        target_distribution = target_series.value_counts(dropna=False).to_dict()

        # Check binary: values must be subset of {0, 1}
        # In python/pandas, int64 0 and 1 or float 0.0 and 1.0 (without fractions)
        is_subset = set(unique_vals).issubset({0, 1, 0.0, 1.0})
        if not is_subset or len(unique_vals) == 0:
            target_is_binary = False
            errors.append(
                f"Target column '{TARGET_COLUMN}' must contain only binary values 0 and 1, "
                f"but found: {unique_vals}"
            )

    is_valid = len(errors) == 0

    return ValidationReport(
        is_valid=is_valid,
        total_rows=total_rows,
        total_columns=total_columns,
        present_columns=present_cols,
        missing_columns=missing_cols,
        unexpected_columns=unexpected_cols,
        missing_values_by_column=missing_by_col,
        total_missing_values=total_missing,
        duplicate_rows_count=duplicate_rows_count,
        target_is_binary=target_is_binary,
        target_unique_values=target_unique_values,
        target_distribution=target_distribution,
        errors=errors,
        warnings=warnings,
    )

