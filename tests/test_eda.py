"""Unit and integration tests for Phase 2 Exploratory Data Analysis (EDA)."""

from pathlib import Path
import pytest
import pandas as pd

from src.data.loader import load_and_validate_data
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


@pytest.fixture(scope="module")
def raw_df():
    """Load and validate the raw AI4I 2020 dataset using Phase 1 pipeline."""
    df, report = load_and_validate_data("data/raw/ai4i2020.csv")
    assert report.is_valid is True
    return df


def test_generate_dataset_overview(raw_df):
    """Verify dataset overview computes correct dimensions, types, and quality metrics."""
    overview = generate_dataset_overview(raw_df)
    assert overview["shape"]["rows"] == 10000
    assert overview["shape"]["columns"] == 14
    assert overview["missing_values"]["total"] == 0
    assert overview["duplicate_rows"] == 0
    assert len(overview["data_types"]) == 14


def test_analyze_target_distribution(raw_df):
    """Verify target distribution counts, percentages, and severe class imbalance ratio."""
    dist = analyze_target_distribution(raw_df)
    assert dist["counts"]["normal_0"] == 9661
    assert dist["counts"]["failed_1"] == 339
    assert dist["percentages"]["normal_0"] == pytest.approx(96.61, abs=0.01)
    assert dist["percentages"]["failed_1"] == pytest.approx(3.39, abs=0.01)
    assert dist["imbalance_ratio"] == "28.5:1"


def test_analyze_numerical_distributions(raw_df):
    """Verify numerical distributions metrics (parametric and non-parametric) for all predictive features."""
    num_stats = analyze_numerical_distributions(raw_df)
    assert set(num_stats.keys()) == set(PREDICTIVE_NUMERICAL_FEATURES)

    for col in PREDICTIVE_NUMERICAL_FEATURES:
        stats = num_stats[col]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "50%_median" in stats
        assert "max" in stats
        assert "skewness" in stats
        assert stats["min"] <= stats["50%_median"] <= stats["max"]


def test_analyze_feature_correlations_leakage_exclusion(raw_df):
    """Verify predictive correlations strictly exclude leakage columns and IDs."""
    corr_matrix, target_corrs = analyze_feature_correlations(raw_df)

    # 1. Target correlations must only contain the 5 predictive features
    assert set(target_corrs.keys()) == set(PREDICTIVE_NUMERICAL_FEATURES)

    # 2. None of the leakage or identifier columns should exist
    for excluded in EXCLUDED_FROM_PREDICTIVE_FEATURES:
        assert excluded not in target_corrs
        assert excluded not in corr_matrix.columns

    # 3. Torque and Tool wear should have the highest positive correlations with failure
    assert target_corrs["Torque [Nm]"] > 0.15
    assert target_corrs["Tool wear [min]"] > 0.10

    # 4. Attempting to pass any leakage column must raise ValueError
    for leakage_col in ["TWF", "HDF", "PWF", "OSF", "RNF", "UDI", "Product ID"]:
        with pytest.raises(ValueError) as exc_info:
            analyze_feature_correlations(raw_df, features=[leakage_col])
        assert "Data leakage violation" in str(exc_info.value)


def test_compare_operating_measurements(raw_df):
    """Verify operating measurement comparison detects elevated torque, tool wear, and thermal load in failures."""
    comp = compare_operating_measurements(raw_df)
    assert set(comp.keys()) == set(PREDICTIVE_NUMERICAL_FEATURES)

    torque_comp = comp["Torque [Nm]"]
    assert torque_comp["failed_mean"] > torque_comp["normal_mean"]
    assert torque_comp["percentage_change"] > 25.0  # Approx +26.6%

    wear_comp = comp["Tool wear [min]"]
    assert wear_comp["failed_mean"] > wear_comp["normal_mean"]
    assert wear_comp["percentage_change"] > 30.0  # Approx +34.8%


def test_generate_eda_visualizations(raw_df, tmp_path):
    """Verify all 4 visualization PNG files are generated with non-zero size."""
    figures_dir = tmp_path / "figures"
    saved_plots = generate_eda_visualizations(raw_df, output_dir=figures_dir)

    expected_plots = [
        "target_distribution",
        "numerical_distributions",
        "correlation_matrix",
        "failed_vs_normal_comparison",
    ]
    for plot_name in expected_plots:
        assert plot_name in saved_plots
        plot_path = saved_plots[plot_name]
        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000  # Non-trivial image file


def test_save_eda_report(raw_df, tmp_path):
    """Verify Markdown and JSON reports are saved and populated."""
    overview = generate_dataset_overview(raw_df)
    target_dist = analyze_target_distribution(raw_df)
    num_stats = analyze_numerical_distributions(raw_df)
    _, target_corrs = analyze_feature_correlations(raw_df)
    comparison = compare_operating_measurements(raw_df)

    md_path, json_path = save_eda_report(
        overview=overview,
        target_dist=target_dist,
        num_stats=num_stats,
        correlations=target_corrs,
        comparison=comparison,
        output_dir=tmp_path,
    )

    assert md_path.exists()
    assert json_path.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "FactoryPulse — Phase 2: Exploratory Data Analysis" in md_text
    assert "Class Imbalance Ratio" in md_text
    assert "Correlation with Machine Failure" in md_text


def test_run_eda_pipeline_end_to_end(tmp_path):
    """Verify end-to-end execution of the full EDA pipeline."""
    results = run_eda_pipeline(
        dataset_path="data/raw/ai4i2020.csv",
        reports_dir=tmp_path,
    )

    assert results["validation_report_status"] == "PASSED"
    assert results["overview"]["shape"]["rows"] == 10000
    assert results["target_distribution"]["counts"]["failed_1"] == 339
    assert Path(results["markdown_report_path"]).exists()
    assert Path(results["json_report_path"]).exists()
    assert (tmp_path / "figures" / "correlation_matrix.png").exists()

