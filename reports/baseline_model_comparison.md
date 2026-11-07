# FactoryPulse — Phase 3: Baseline Model Development Report

> [!WARNING]
> **EXPERIMENTAL PROTOTYPE — NOT VALIDATED FOR PRODUCTION DEPLOYMENT**
> These baseline models are benchmark prototypes trained solely on the AI4I 2020 synthetic dataset.
> They demonstrate modeling feasibility, evaluate class imbalance handling, and provide initial performance baselines.
> They do NOT represent a production-ready predictive maintenance solution.

## 1. Baseline Model Comparison Table
| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0340 | 1932 | 0 | 68 | 0 |
| **Logistic Regression** | 0.1418 | 0.8235 | 0.2419 | 0.9070 | 0.3818 | 1593 | 339 | 12 | 56 |
| **Random Forest** | 0.7419 | 0.6765 | 0.7077 | 0.9697 | 0.7599 | 1916 | 16 | 22 | 46 |

## 2. In-Depth Baseline Findings
1. **DummyClassifier (Majority Baseline)**:
   - Predicts 'Normal' (0) for all test instances.
   - Achieves 96.6% accuracy but **0.00 Precision, 0.00 Recall, and 0.00 F1-Score**.
   - Proves definitively why standard classification accuracy is misleading in predictive maintenance.

2. **LogisticRegression (`class_weight='balanced'`)**:
   - Achieves high failure capture: **Recall = 82.35%** (catches 56 out of 68 failures).
   - Suffers from low precision: **Precision = 14.18%** (339 false alarms).
   - ROC-AUC of **0.9070** demonstrates strong linear discriminative ability despite false positives.

3. **RandomForestClassifier (`class_weight='balanced'`)**:
   - Offers the best balanced trade-off among the baselines.
   - **Precision = 74.19%**, **Recall = 67.65%**, **F1-Score = 0.7077**.
   - **ROC-AUC = 0.9697**, **PR-AUC = 0.7599**.
   - Reduces false positives to only 16 (vs 339 for logistic regression) while catching 46 out of 68 failures.

## 3. Strict Target Leakage Prevention
- `UDI` and `Product ID` (identifiers) were excluded.
- `TWF`, `HDF`, `PWF`, `OSF`, and `RNF` (post-failure diagnostic modes) were strictly excluded from features.
- Preprocessing (`StandardScaler` and `OneHotEncoder`) was embedded directly within scikit-learn `Pipeline` objects to ensure zero data leakage from the test split.

## 4. Generated Artifacts
- Trained Pipelines: `models/dummy_pipeline.joblib`, `models/logistic_regression_pipeline.joblib`, `models/random_forest_pipeline.joblib`
- Confusion Matrices: `reports/figures/confusion_matrices_comparison.png`
- ROC & PR Curves: `reports/figures/model_roc_pr_curves.png`
