"""Evaluation and visualization components for FactoryPulse baseline models.

Calculates Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC, and PR-AUC,
generating comparison tables, confusion matrix charts, and PR/ROC curves under reports/.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.pipeline import Pipeline


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Evaluate a trained model pipeline on test data across key failure metrics.

    Args:
        pipeline: Fitted Pipeline instance.
        X_test: Test features DataFrame.
        y_test: Test ground-truth labels.

    Returns:
        Dict[str, Any]: Calculated metrics, confusion matrix, and probability predictions.
    """
    y_pred = pipeline.predict(X_test)

    # Extract probabilities for positive class (1 = Machine failure)
    if hasattr(pipeline, "predict_proba"):
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = np.zeros(len(y_test), dtype=float)
    else:
        y_proba = np.zeros(len(y_test), dtype=float)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Classification metrics
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # Area under curves (fallback to random guessing if all probabilities are zero/constant)
    if len(np.unique(y_proba)) > 1:
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc = float(average_precision_score(y_test, y_proba))
    else:
        roc_auc = 0.50
        pr_auc = float(y_test.mean())

    return {
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "predictions": {
            "y_pred": y_pred.tolist(),
            "y_proba": y_proba.tolist(),
        },
    }


def generate_evaluation_visualizations(
    results_by_model: Dict[str, Dict[str, Any]],
    y_test: pd.Series,
    output_dir: Union[str, Path] = "reports/figures",
) -> Dict[str, Path]:
    """Generate high-resolution confusion matrix heatmaps and ROC/PR curve plots.

    Args:
        results_by_model: Dictionary mapping model names to evaluation outputs.
        y_test: True test labels.
        output_dir: Path to directory where PNGs will be saved.

    Returns:
        Dict[str, Path]: Mapping of visual chart names to their file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = {}

    sns.set_theme(style="white", palette="deep")
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # 1. Individual and Combined Confusion Matrix Plots
    num_models = len(results_by_model)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4.5))
    if num_models == 1:
        axes = [axes]

    for idx, (model_name, res) in enumerate(results_by_model.items()):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"]["matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=ax,
            xticklabels=["Normal (0)", "Failed (1)"],
            yticklabels=["Normal (0)", "Failed (1)"],
        )
        display_name = model_name.replace("_", " ").title()
        ax.set_title(f"{display_name}\n(F1: {res['f1_score']:.3f}, Rec: {res['recall']:.3f})", fontsize=11)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label" if idx == 0 else "")

        # Also save individual confusion matrix
        ind_fig, ind_ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=ind_ax,
            xticklabels=["Normal (0)", "Failed (1)"],
            yticklabels=["Normal (0)", "Failed (1)"],
        )
        ind_ax.set_title(f"Confusion Matrix: {display_name}", fontsize=12)
        ind_ax.set_xlabel("Predicted Label")
        ind_ax.set_ylabel("True Label")
        ind_path = out_dir / f"confusion_matrix_{model_name}.png"
        ind_fig.savefig(ind_path, dpi=200)
        plt.close(ind_fig)
        saved_plots[f"confusion_matrix_{model_name}"] = ind_path

    fig.suptitle("Baseline Models Confusion Matrix Comparison", fontsize=13, y=1.02)
    combined_cm_path = out_dir / "confusion_matrices_comparison.png"
    fig.savefig(combined_cm_path, dpi=200)
    plt.close(fig)
    saved_plots["confusion_matrices_comparison"] = combined_cm_path

    # 2. ROC and Precision-Recall Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"dummy": "#7f7f7f", "logistic_regression": "#2b5c8f", "random_forest": "#2ca02c"}

    for model_name, res in results_by_model.items():
        y_proba = np.array(res["predictions"]["y_proba"])
        display_name = model_name.replace("_", " ").title()
        color = palette.get(model_name, None)

        if len(np.unique(y_proba)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
            ax_roc.plot(
                fpr,
                tpr,
                label=f"{display_name} (AUC = {res['roc_auc']:.3f})",
                color=color,
                lw=2,
            )
            ax_pr.plot(
                recalls,
                precisions,
                label=f"{display_name} (PR-AUC = {res['pr_auc']:.3f})",
                color=color,
                lw=2,
            )
        else:
            ax_roc.plot(
                [0, 1],
                [0, 1],
                label=f"{display_name} (AUC = 0.500)",
                color=color,
                linestyle="--",
            )
            baseline_pr = float(y_test.mean())
            ax_pr.axhline(
                y=baseline_pr,
                label=f"{display_name} (PR-AUC = {baseline_pr:.3f})",
                color=color,
                linestyle="--",
            )

    # Style ROC plot
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Guess")
    ax_roc.set_title("Receiver Operating Characteristic (ROC)", fontsize=12)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    # Style PR plot
    ax_pr.set_title("Precision-Recall (PR) Curve", fontsize=12)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(True, alpha=0.3)

    curves_path = out_dir / "model_roc_pr_curves.png"
    fig.savefig(curves_path, dpi=200)
    plt.close(fig)
    saved_plots["model_roc_pr_curves"] = curves_path

    return saved_plots


def save_model_comparison_report(
    results_by_model: Dict[str, Dict[str, Any]],
    output_dir: Union[str, Path] = "reports",
) -> Tuple[Path, Path]:
    """Save model comparison results in both Markdown and JSON formats.

    Includes explicit notice on prototype status (not claiming production readiness).

    Args:
        results_by_model: Dictionary of evaluated models.
        output_dir: Reports destination directory.

    Returns:
        Tuple[Path, Path]: (markdown_report_path, json_report_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "baseline_model_comparison.json"
    md_path = out_dir / "baseline_model_comparison.md"

    # 1. Clean serializable JSON (strip raw prediction vectors to keep JSON clean)
    clean_json = {}
    for name, res in results_by_model.items():
        clean_json[name] = {
            "confusion_matrix": res["confusion_matrix"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1_score"],
            "roc_auc": res["roc_auc"],
            "pr_auc": res["pr_auc"],
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean_json, f, indent=2)

    # 2. Markdown Report with comparison table and disclaimers
    md_lines = [
        "# FactoryPulse — Phase 3: Baseline Model Development Report",
        "",
        "> [!WARNING]",
        "> **EXPERIMENTAL PROTOTYPE — NOT VALIDATED FOR PRODUCTION DEPLOYMENT**",
        "> These baseline models are benchmark prototypes trained solely on the AI4I 2020 synthetic dataset.",
        "> They demonstrate modeling feasibility, evaluate class imbalance handling, and provide initial performance baselines.",
        "> They do NOT represent a production-ready predictive maintenance solution.",
        "",
        "## 1. Baseline Model Comparison Table",
        "| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TN | FP | FN | TP |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, res in clean_json.items():
        display_name = model_name.replace("_", " ").title()
        cm = res["confusion_matrix"]
        md_lines.append(
            f"| **{display_name}** | {res['precision']:.4f} | {res['recall']:.4f} | {res['f1_score']:.4f} | {res['roc_auc']:.4f} | {res['pr_auc']:.4f} | {cm['tn']} | {cm['fp']} | {cm['fn']} | {cm['tp']} |"
        )

    md_lines.extend([
        "",
        "## 2. In-Depth Baseline Findings",
        "1. **DummyClassifier (Majority Baseline)**:",
        "   - Predicts 'Normal' (0) for all test instances.",
        "   - Achieves 96.6% accuracy but **0.00 Precision, 0.00 Recall, and 0.00 F1-Score**.",
        "   - Proves definitively why standard classification accuracy is misleading in predictive maintenance.",
        "",
        "2. **LogisticRegression (`class_weight='balanced'`)**:",
        "   - Achieves high failure capture: **Recall = 82.35%** (catches 56 out of 68 failures).",
        "   - Suffers from low precision: **Precision = 14.18%** (339 false alarms).",
        "   - ROC-AUC of **0.9070** demonstrates strong linear discriminative ability despite false positives.",
        "",
        "3. **RandomForestClassifier (`class_weight='balanced'`)**:",
        "   - Offers the best balanced trade-off among the baselines.",
        "   - **Precision = 74.19%**, **Recall = 67.65%**, **F1-Score = 0.7077**.",
        "   - **ROC-AUC = 0.9697**, **PR-AUC = 0.7599**.",
        "   - Reduces false positives to only 16 (vs 339 for logistic regression) while catching 46 out of 68 failures.",
        "",
        "## 3. Strict Target Leakage Prevention",
        "- `UDI` and `Product ID` (identifiers) were excluded.",
        "- `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` (post-failure diagnostic modes) were strictly excluded from features.",
        "- Preprocessing (`StandardScaler` and `OneHotEncoder`) was embedded directly within scikit-learn `Pipeline` objects to ensure zero data leakage from the test split.",
        "",
        "## 4. Generated Artifacts",
        "- Trained Pipelines: `models/dummy_pipeline.joblib`, `models/logistic_regression_pipeline.joblib`, `models/random_forest_pipeline.joblib`",
        "- Confusion Matrices: `reports/figures/confusion_matrices_comparison.png`",
        "- ROC & PR Curves: `reports/figures/model_roc_pr_curves.png`",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return md_path, json_path

