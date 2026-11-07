# FactoryPulse — Phase 2: Exploratory Data Analysis (EDA) Report

## 1. Dataset Overview
- **Shape**: 10,000 rows × 14 columns
- **Missing Values**: 0 missing cells detected
- **Duplicate Rows**: 0 duplicate records detected

## 2. Machine Failure Distribution
- **Normal Operating Units (0)**: 9,661 (96.61%)
- **Failed Units (1)**: 339 (3.39%)
- **Class Imbalance Ratio**: 28.5:1 (Normal : Failed)

> [!IMPORTANT]
> **Severe Imbalance Notice**: Machine failures represent only 3.39% of the dataset.
> Accuracy is not a suitable metric because a trivial model predicting '0' achieves 96.61% accuracy.
> Evaluation in Phase 3 must prioritize **PR-AUC (Precision-Recall AUC)**, **ROC-AUC**, and **F1-Score / Recall**.

## 3. Operational Sensor Distributions
| Sensor Feature | Mean | Std | Min | 25% | Median (50%) | 75% | Max | Skewness |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Air temperature [K]** | 300.00 | 2.00 | 295.30 | 298.30 | 300.10 | 301.50 | 304.50 | +0.11 |
| **Process temperature [K]** | 310.01 | 1.48 | 305.70 | 308.80 | 310.10 | 311.10 | 313.80 | +0.02 |
| **Rotational speed [rpm]** | 1538.78 | 179.28 | 1168.00 | 1423.00 | 1503.00 | 1612.00 | 2886.00 | +1.99 |
| **Torque [Nm]** | 39.99 | 9.97 | 3.80 | 33.20 | 40.10 | 46.80 | 76.60 | -0.01 |
| **Tool wear [min]** | 107.95 | 63.65 | 0.00 | 53.00 | 108.00 | 162.00 | 253.00 | +0.03 |

## 4. Correlation with Machine Failure (Target Leakage Prevented)
> [!NOTE]
> `UDI` and `Product ID` (non-predictive identifiers) as well as `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`
> (post-failure diagnostic modes) have been **strictly excluded** from predictive correlation to prevent target leakage.

| Predictive Feature | Correlation with Machine Failure | Direction & Interpretation |
| :--- | :---: | :--- |
| **Torque [Nm]** | `+0.1913` | Strong positive correlation (higher values increase failure risk) |
| **Tool wear [min]** | `+0.1054` | Moderate positive correlation |
| **Air temperature [K]** | `+0.0826` | Moderate positive correlation |
| **Process temperature [K]** | `+0.0359` | Weak linear correlation |
| **Rotational speed [rpm]** | `-0.0442` | Weak linear correlation |

## 5. Comparative Analysis: Normal (0) vs Failed (1) Machines
| Measurement | Normal Mean (0) | Failed Mean (1) | Absolute Difference | Relative % Shift |
| :--- | :---: | :---: | :---: | :---: |
| **Air temperature [K]** | 299.97 | 300.89 | +0.91 | **+0.30%** |
| **Process temperature [K]** | 310.00 | 310.29 | +0.29 | **+0.10%** |
| **Rotational speed [rpm]** | 1540.26 | 1496.49 | -43.77 | **-2.84%** |
| **Torque [Nm]** | 39.63 | 50.17 | +10.54 | **+26.59%** |
| **Tool wear [min]** | 106.69 | 143.78 | +37.09 | **+34.76%** |

## 6. Key Engineering Insights
1. **Torque Surge (+26.6%)**: Failed machines show significantly higher average torque (50.17 Nm vs 39.63 Nm). High cutting load is a primary failure indicator.
2. **Tool Wear Accumulation (+34.8%)**: Failed tools exhibit an average wear of 143.78 minutes compared to 106.69 minutes for healthy machines, confirming accelerated fatigue.
3. **Thermal Load (+0.91 K air temp)**: Ambient and process temperatures are elevated during failures, impairing natural heat dissipation.
4. **Rotational Speed Instability**: While the mean speed shifts modestly (-2.8%), the variance jumps dramatically (std 384.9 rpm for failures vs 167.4 rpm for normal), reflecting motor stalling or severe speed drops.

## 7. Generated Visualizations
- `reports/figures/target_distribution.png`: Bar chart of class counts and imbalance ratio.
- `reports/figures/numerical_distributions.png`: Histograms with KDE for operational sensor signals.
- `reports/figures/correlation_matrix.png`: Heatmap showing predictive correlation with failure.
- `reports/figures/failed_vs_normal_comparison.png`: Boxplots highlighting elevated torque and tool wear.
