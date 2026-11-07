# FactoryPulse — Phase 3.5: Random Forest Threshold Analysis Report

> [!WARNING]
> **OPERATIONAL DISCLAIMER: NOT VALIDATED FOR FACTORY PRODUCTION**
> The recommended threshold is evaluated on the synthetic AI4I 2020 dataset benchmark.
> In an actual industrial manufacturing environment, the operational threshold must be
> validated against true equipment downtime costs, repair expenses, and false-alarm triage capacity.

## 1. Executive Summary & Recommendation
- **Existing Default Threshold**: `0.50` (Decision cutoff used in baseline evaluation)
- **Recommended Threshold**: `0.45` (Selected based on 5-Fold Stratified Cross-Validation on training data)
- **Trade-off Rationale**: Lowering the cutoff from 0.50 to 0.45 increases machine failure capture rate (Recall) while maintaining acceptable precision.

## 2. Cross-Validation Threshold Comparison Table (Training Split Only, 8,000 Samples)
| Threshold | Precision | Recall | F1-Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.10` | 0.2742 | 0.9299 | 0.4235 | 252 | 667 | 19 | 7062 |
| `0.20` | 0.3807 | 0.8598 | 0.5277 | 233 | 379 | 38 | 7350 |
| `0.25` | 0.4382 | 0.8376 | 0.5754 | 227 | 291 | 44 | 7438 |
| `0.30` | 0.5035 | 0.7970 | 0.6171 | 216 | 213 | 55 | 7516 |
| `0.35` | 0.5622 | 0.7675 | 0.6490 | 208 | 162 | 63 | 7567 |
| `0.40` | 0.6037 | 0.7306 | 0.6611 | 198 | 130 | 73 | 7599 |
| `0.45` **(Recommended)** | 0.6631 | 0.6827 | 0.6727 | 185 | 94 | 86 | 7635 |
| `0.50` | 0.6992 | 0.6347 | 0.6654 | 172 | 74 | 99 | 7655 |
| `0.55` | 0.7561 | 0.5720 | 0.6513 | 155 | 50 | 116 | 7679 |
| `0.60` | 0.8046 | 0.5166 | 0.6292 | 140 | 34 | 131 | 7695 |
| `0.65` | 0.8357 | 0.4317 | 0.5693 | 117 | 23 | 154 | 7706 |
| `0.70` | 0.8699 | 0.3948 | 0.5431 | 107 | 16 | 164 | 7713 |
| `0.75` | 0.9167 | 0.2841 | 0.4338 | 77 | 7 | 194 | 7722 |
| `0.80` | 0.9508 | 0.2140 | 0.3494 | 58 | 3 | 213 | 7726 |

## 3. Single-Pass Evaluation on Untouched Test Set (2,000 Samples)
| Configuration | Threshold | Precision | Recall | F1-Score | TP (Caught) | FP (False Alarms) | FN (Missed) | TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Baseline** | `0.50` | 0.7460 | 0.6912 | 0.7176 | 47 | 16 | 21 | 1916 |
| **Selected Threshold** | `0.45` | 0.6575 | 0.7059 | 0.6809 | 48 | 25 | 20 | 1907 |
| **Net Change (Delta)** | `-0.05` | `-0.0885` | **`+0.0147`** | `-0.0367` | **+1** | +9 | **-1** | -9 |

## 4. Precision-Recall Trade-off Insights
1. **Failure Cost Asymmetry**: In industrial maintenance, a False Negative (undetected breakdown) typically costs 10x-50x more than a False Positive (scheduled technician check).
2. **Safety Margin**: Shifting the decision threshold to `0.45` allows the maintenance team to intercept **48 out of 68 test failures (70.6%)** compared to 47 (69.1%) at default cutoff.
3. **Operational Burden**: The trade-off is an increase in false alarms, which must be factored into factory technician schedules.

## 5. Generated Artifacts
- `reports/figures/rf_threshold_vs_metrics.png`: Precision, Recall, and F1 curves across thresholds.
- `reports/figures/rf_confusion_matrix_threshold_comparison.png`: Side-by-side confusion matrix comparison on test set.
- `reports/rf_threshold_analysis.json`: Full machine-readable numerical dataset.
