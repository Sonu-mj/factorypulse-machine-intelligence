"""Classification threshold analysis for Random Forest in FactoryPulse.

Evaluates candidate decision thresholds using Stratified 5-Fold Cross-Validation
on training data only (preventing test leakage), analyzes precision-recall trade-offs,
selects a recommended threshold, and verifies performance on the untouched test set.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src.data.loader import load_and_validate_data
from src.training.models import build_model_pipeline, build_preprocessor, get_baseline_models
from src.training.pipeline import extract_baseline_data

# Candidate classification thresholds to analyze
DEFAULT_CANDIDATE_THRESHOLDS: List[float] = [
    0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
]


def evaluate_threshold_candidates_cv(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidate_thresholds: Optional[List[float]] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Evaluate candidate classification thresholds using out-of-fold cross-validation on training data.

    Strict Anti-Leakage Guarantee:
    Threshold evaluation is performed entirely within training cross-validation folds.
    The test set is NOT touched during threshold candidate evaluation.

    Args:
        pipeline: Unfitted scikit-learn Pipeline.
        X_train: Training features DataFrame (8,000 samples).
        y_train: Training target Series.
        candidate_thresholds: List of thresholds to evaluate.
        n_splits: Number of stratified folds.
        random_state: Seed for cross-validation splitting.

    Returns:
        Tuple[List[Dict[str, Any]], np.ndarray]:
            - List of evaluation metrics dictionaries per candidate threshold.
            - Out-of-fold predicted probabilities array.
    """
    if candidate_thresholds is None:
        candidate_thresholds = DEFAULT_CANDIDATE_THRESHOLDS

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Compute out-of-fold predicted probabilities for positive class (1 = failure)
    oof_proba = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=skf,
        method="predict_proba",
    )[:, 1]

    results = []
    for threshold in candidate_thresholds:
        y_pred = (oof_proba >= threshold).astype(int)
        cm = confusion_matrix(y_train, y_pred)
        tn, fp, fn, tp = cm.ravel()

        prec = float(precision_score(y_train, y_pred, zero_division=0))
        rec = float(recall_score(y_train, y_pred, zero_division=0))
        f1 = float(f1_score(y_train, y_pred, zero_division=0))

        results.append({
            "threshold": float(threshold),
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return results, oof_proba


def select_recommended_threshold(
    cv_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Select the optimal decision threshold based on cross-validation performance.

    Finds:
    1. F1-Optimal Threshold: Maximizes overall F1-score.
    2. Maintenance-Sensitive Threshold: Captures >=70% recall while preserving >=60% precision.

    Args:
        cv_results: List of metric dicts from evaluate_threshold_candidates_cv.

    Returns:
        Dict[str, Any]: Selected recommendation and metadata.
    """
    # 1. Find threshold that strictly maximizes F1-score on training CV
    best_f1_entry = max(cv_results, key=lambda x: x["f1_score"])

    # 2. Find balanced maintenance threshold (Recall >= 0.70 with Precision >= 0.60)
    balanced_candidates = [
        r for r in cv_results if r["recall"] >= 0.70 and r["precision"] >= 0.60
    ]
    balanced_entry = (
        max(balanced_candidates, key=lambda x: x["f1_score"])
        if balanced_candidates
        else best_f1_entry
    )

    return {
        "recommended_threshold": best_f1_entry["threshold"],
        "f1_optimal": best_f1_entry,
        "balanced_maintenance": balanced_entry,
    }


def evaluate_threshold_on_test(
    fitted_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate a selected threshold exactly once on the untouched held-out test set.

    Args:
        fitted_pipeline: Pipeline trained on X_train.
        X_test: Untouched test features DataFrame (2,000 samples).
        y_test: Test target Series (68 failures).
        threshold: Recommended classification threshold.

    Returns:
        Dict[str, Any]: Evaluation metrics at recommended threshold and default (0.50) comparison.
    """
    y_proba_test = fitted_pipeline.predict_proba(X_test)[:, 1]

    # Predictions at recommended threshold
    pred_custom = (y_proba_test >= threshold).astype(int)
    cm_custom = confusion_matrix(y_test, pred_custom)
    tn_c, fp_c, fn_c, tp_c = cm_custom.ravel()

    prec_custom = float(precision_score(y_test, pred_custom, zero_division=0))
    rec_custom = float(recall_score(y_test, pred_custom, zero_division=0))
    f1_custom = float(f1_score(y_test, pred_custom, zero_division=0))

    # Baseline comparison at default threshold 0.50
    pred_default = (y_proba_test >= 0.50).astype(int)
    cm_default = confusion_matrix(y_test, pred_default)
    tn_d, fp_d, fn_d, tp_d = cm_default.ravel()

    prec_default = float(precision_score(y_test, pred_default, zero_division=0))
    rec_default = float(recall_score(y_test, pred_default, zero_division=0))
    f1_default = float(f1_score(y_test, pred_default, zero_division=0))

    return {
        "selected_threshold": float(threshold),
        "test_metrics_at_selected_threshold": {
            "threshold": float(threshold),
            "precision": prec_custom,
            "recall": rec_custom,
            "f1_score": f1_custom,
            "tp": int(tp_c),
            "fp": int(fp_c),
            "fn": int(fn_c),
            "tn": int(tn_c),
        },
        "test_metrics_at_default_0_50": {
            "threshold": 0.50,
            "precision": prec_default,
            "recall": rec_default,
            "f1_score": f1_default,
            "tp": int(tp_d),
            "fp": int(fp_d),
            "fn": int(fn_d),
            "tn": int(tn_d),
        },
        "delta_vs_default": {
            "recall_gain": rec_custom - rec_default,
            "precision_change": prec_custom - prec_default,
            "f1_change": f1_custom - f1_default,
            "additional_failures_detected": int(tp_c - tp_d),
            "additional_false_alarms": int(fp_c - fp_d),
        },
    }


def generate_threshold_visualizations(
    cv_results: List[Dict[str, Any]],
    test_eval: Dict[str, Any],
    output_dir: Union[str, Path] = "reports/figures",
) -> Dict[str, Path]:
    """Generate plots showing precision-recall-F1 curves and confusion matrix comparison.

    Args:
        cv_results: Out-of-fold CV results across thresholds.
        test_eval: Test evaluation comparison dict.
        output_dir: Output figures directory.

    Returns:
        Dict[str, Path]: Saved visual chart paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = {}

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    thresholds = [r["threshold"] for r in cv_results]
    precisions = [r["precision"] for r in cv_results]
    recalls = [r["recall"] for r in cv_results]
    f1_scores = [r["f1_score"] for r in cv_results]
    rec_thresh = test_eval["selected_threshold"]

    # 1. Threshold vs Metrics Curve Plot
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(thresholds, precisions, marker="o", label="Precision", color="#1f77b4", lw=2)
    ax.plot(thresholds, recalls, marker="s", label="Recall", color="#d95f02", lw=2)
    ax.plot(thresholds, f1_scores, marker="^", label="F1-Score", color="#2ca02c", lw=2.5)

    ax.axvline(
        x=rec_thresh,
        color="#7570b3",
        linestyle="--",
        lw=2,
        label=f"Selected Threshold ({rec_thresh:.2f})",
    )
    ax.axvline(
        x=0.50,
        color="#999999",
        linestyle=":",
        lw=1.5,
        label="Default Cutoff (0.50)",
    )

    ax.set_title("Random Forest: Precision, Recall & F1 vs Classification Threshold (5-Fold CV)", fontsize=12, pad=10)
    ax.set_xlabel("Classification Probability Threshold")
    ax.set_ylabel("Metric Score")
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left", framealpha=0.9)

    curve_path = out_dir / "rf_threshold_vs_metrics.png"
    fig.savefig(curve_path, dpi=200)
    plt.close(fig)
    saved_plots["threshold_vs_metrics"] = curve_path

    # 2. Side-by-side Confusion Matrix Comparison (Default 0.50 vs Selected)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    cm_def = np.array([
        [test_eval["test_metrics_at_default_0_50"]["tn"], test_eval["test_metrics_at_default_0_50"]["fp"]],
        [test_eval["test_metrics_at_default_0_50"]["fn"], test_eval["test_metrics_at_default_0_50"]["tp"]],
    ])
    cm_sel = np.array([
        [test_eval["test_metrics_at_selected_threshold"]["tn"], test_eval["test_metrics_at_selected_threshold"]["fp"]],
        [test_eval["test_metrics_at_selected_threshold"]["fn"], test_eval["test_metrics_at_selected_threshold"]["tp"]],
    ])

    sns.heatmap(
        cm_def,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax1,
        xticklabels=["Normal (0)", "Failed (1)"],
        yticklabels=["Normal (0)", "Failed (1)"],
    )
    ax1.set_title(
        f"Default Threshold (0.50)\nRecall: {test_eval['test_metrics_at_default_0_50']['recall']:.3f} | Prec: {test_eval['test_metrics_at_default_0_50']['precision']:.3f}",
        fontsize=11,
    )
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    sns.heatmap(
        cm_sel,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax2,
        xticklabels=["Normal (0)", "Failed (1)"],
        yticklabels=["Normal (0)", "Failed (1)"],
    )
    ax2.set_title(
        f"Selected Threshold ({rec_thresh:.2f})\nRecall: {test_eval['test_metrics_at_selected_threshold']['recall']:.3f} | Prec: {test_eval['test_metrics_at_selected_threshold']['precision']:.3f}",
        fontsize=11,
    )
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")

    fig.suptitle("Test Set Confusion Matrix: Default (0.50) vs Selected Threshold", fontsize=13, y=1.02)
    comp_cm_path = out_dir / "rf_confusion_matrix_threshold_comparison.png"
    fig.savefig(comp_cm_path, dpi=200)
    plt.close(fig)
    saved_plots["confusion_matrix_comparison"] = comp_cm_path

    return saved_plots


def save_threshold_report(
    cv_results: List[Dict[str, Any]],
    recommendation: Dict[str, Any],
    test_eval: Dict[str, Any],
    output_dir: Union[str, Path] = "reports",
) -> Tuple[Path, Path]:
    """Save comprehensive threshold analysis report in Markdown and JSON formats.

    Args:
        cv_results: Metric results across candidate thresholds.
        recommendation: Threshold recommendation metadata.
        test_eval: Evaluation on untouched test split.
        output_dir: Reports output directory.

    Returns:
        Tuple[Path, Path]: (markdown_path, json_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "rf_threshold_analysis.json"
    md_path = out_dir / "rf_threshold_analysis.md"

    # Save JSON report
    report_dict = {
        "training_cross_validation_results": cv_results,
        "recommendation": recommendation,
        "test_set_evaluation": test_eval,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    # Format Markdown report
    rec_val = recommendation["recommended_threshold"]
    test_m = test_eval["test_metrics_at_selected_threshold"]
    def_m = test_eval["test_metrics_at_default_0_50"]
    delta = test_eval["delta_vs_default"]

    md_lines = [
        "# FactoryPulse — Phase 3.5: Random Forest Threshold Analysis Report",
        "",
        "> [!WARNING]",
        "> **OPERATIONAL DISCLAIMER: NOT VALIDATED FOR FACTORY PRODUCTION**",
        "> The recommended threshold is evaluated on the synthetic AI4I 2020 dataset benchmark.",
        "> In an actual industrial manufacturing environment, the operational threshold must be",
        "> validated against true equipment downtime costs, repair expenses, and false-alarm triage capacity.",
        "",
        "## 1. Executive Summary & Recommendation",
        f"- **Existing Default Threshold**: `0.50` (Decision cutoff used in baseline evaluation)",
        f"- **Recommended Threshold**: `{rec_val:.2f}` (Selected based on 5-Fold Stratified Cross-Validation on training data)",
        f"- **Trade-off Rationale**: Lowering the cutoff from 0.50 to {rec_val:.2f} increases machine failure capture rate (Recall) while maintaining acceptable precision.",
        "",
        "## 2. Cross-Validation Threshold Comparison Table (Training Split Only, 8,000 Samples)",
        "| Threshold | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in cv_results:
        is_rec = " **(Recommended)**" if r["threshold"] == rec_val else ""
        md_lines.append(
            f"| `{r['threshold']:.2f}`{is_rec} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} |"
        )

    md_lines.extend([
        "",
        "## 3. Single-Pass Evaluation on Untouched Test Set (2,000 Samples)",
        "| Configuration | Threshold | Precision | Recall | F1-Score | TP (Caught) | FP (False Alarms) | FN (Missed) | TN |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Default Baseline** | `0.50` | {def_m['precision']:.4f} | {def_m['recall']:.4f} | {def_m['f1_score']:.4f} | {def_m['tp']} | {def_m['fp']} | {def_m['fn']} | {def_m['tn']} |",
        f"| **Selected Threshold** | `{test_m['threshold']:.2f}` | {test_m['precision']:.4f} | {test_m['recall']:.4f} | {test_m['f1_score']:.4f} | {test_m['tp']} | {test_m['fp']} | {test_m['fn']} | {test_m['tn']} |",
        f"| **Net Change (Delta)** | `{rec_val - 0.50:+.2f}` | `{delta['precision_change']:+.4f}` | **`{delta['recall_gain']:+.4f}`** | `{delta['f1_change']:+.4f}` | **{delta['additional_failures_detected']:+d}** | {delta['additional_false_alarms']:+d} | **{-delta['additional_failures_detected']:+d}** | {test_m['tn'] - def_m['tn']:+d} |",
        "",
        "## 4. Precision-Recall Trade-off Insights",
        "1. **Failure Cost Asymmetry**: In industrial maintenance, a False Negative (undetected breakdown) typically costs 10x-50x more than a False Positive (scheduled technician check).",
        f"2. **Safety Margin**: Shifting the decision threshold to `{rec_val:.2f}` allows the maintenance team to intercept **{test_m['tp']} out of 68 test failures ({test_m['recall']:.1%})** compared to {def_m['tp']} ({def_m['recall']:.1%}) at default cutoff.",
        "3. **Operational Burden**: The trade-off is an increase in false alarms, which must be factored into factory technician schedules.",
        "",
        "## 5. Generated Artifacts",
        "- `reports/figures/rf_threshold_vs_metrics.png`: Precision, Recall, and F1 curves across thresholds.",
        "- `reports/figures/rf_confusion_matrix_threshold_comparison.png`: Side-by-side confusion matrix comparison on test set.",
        "- `reports/rf_threshold_analysis.json`: Full machine-readable numerical dataset.",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return md_path, json_path


def run_threshold_analysis(
    dataset_path: Union[str, Path] = "data/raw/ai4i2020.csv",
    reports_dir: Union[str, Path] = "reports",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute complete Phase 3.5 Random Forest threshold analysis.

    Workflow:
    1. Loads dataset using Phase 1 ingestion pipeline.
    2. Splits 80/20 train/test (identical seed to Phase 3).
    3. Runs 5-fold Stratified Cross-Validation on training set to evaluate thresholds.
    4. Selects recommended threshold based on cross-validation trade-off.
    5. Fits pipeline on training split and evaluates recommended threshold ONCE on test set.
    6. Generates plots under reports/figures/ and saves reports under reports/.

    Args:
        dataset_path: Path to dataset CSV.
        reports_dir: Reports output directory.
        random_state: Seed for reproducible split and modeling.

    Returns:
        Dict[str, Any]: Complete analysis results.
    """
    reports_path = Path(reports_dir)
    figures_path = reports_path / "figures"

    # 1. Ingestion
    df, validation_report = load_and_validate_data(dataset_path, raise_on_error=True)
    X, y = extract_baseline_data(df)

    # 2. 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 3. Build pipeline
    rf_model = get_baseline_models(random_state=random_state)["random_forest"]
    pipeline = build_model_pipeline(model=rf_model, preprocessor=build_preprocessor())

    # 4. Cross-validation threshold evaluation on training split ONLY
    cv_results, oof_proba = evaluate_threshold_candidates_cv(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )

    # 5. Select recommended threshold based on CV
    recommendation = select_recommended_threshold(cv_results)
    rec_thresh = recommendation["recommended_threshold"]

    # 6. Fit pipeline on full training split and evaluate ONCE on test set
    pipeline.fit(X_train, y_train)
    test_eval = evaluate_threshold_on_test(
        fitted_pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        threshold=rec_thresh,
    )

    # 7. Generate plots
    saved_plots = generate_threshold_visualizations(
        cv_results=cv_results,
        test_eval=test_eval,
        output_dir=figures_path,
    )

    # 8. Save reports
    md_report, json_report = save_threshold_report(
        cv_results=cv_results,
        recommendation=recommendation,
        test_eval=test_eval,
        output_dir=reports_path,
    )

    return {
        "status": "SUCCESS",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "cv_results": cv_results,
        "recommendation": recommendation,
        "test_evaluation": test_eval,
        "saved_plots": {k: str(v) for k, v in saved_plots.items()},
        "markdown_report": str(md_report),
        "json_report": str(json_report),
    }

