# Telco Customer Churn MLOps Pipeline

An end-to-end MLOps architecture designed to predict, evaluate, and track customer churn for telecommunications data. This project emphasizes reproducible data engineering, strict class-imbalance mitigation, and business-aligned model evaluation via MLflow.

## 1. Problem

Customer churn is a critical revenue drain in the telecommunications industry. The objective is to identify subscribers at high risk of churning before they cancel their service, allowing the business to intervene with targeted retention offers. Because acquiring a new customer is significantly more expensive than retaining an existing one, the cost of failing to catch a churner (False Negative) heavily outweighs the cost of a wasted retention incentive (False Positive).

## 2. Data

* **Source:** Processed Telco customer dataset (`data/processed/telco_clean.parquet`).

* **Class Distribution:** Imbalanced target (\~26.5% churn rate, \~73.5% non-churn).

* **Data Quality Guards:** Automated validation checks in `data.py` enforcing strict row bounds, type safety, and explicit expected churn rate guardrails.

## 3. Approach

* **Pipeline Encapsulation:** Preprocessing steps (scaling and encoding) are strictly coupled inside scikit-learn `Pipeline` objects to eliminate data leakage during cross-validation. (Note: Missing value handling for numerical columns like `TotalCharges` is cleanly handled upstream in `data.py`).

* **Stratified K-Fold CV:** Implemented 5-fold stratified cross-validation (`StratifiedKFold`) to preserve the 26.5% minority class ratio across every fold and yield stable out-of-fold performance estimates.

* **Class Imbalance Strategy:** Applied `class_weight='balanced'` to inversely weight classes for Logistic Regression and Random Forest, forcing models to prioritize minority 'Yes' churners.

  * *Note on Comparability:* `class_weight='balanced'` was applied to LR and RF, but not to Gradient Boosting (`GradientBoostingClassifier` does not natively expose `class_weight`, and explicit sample-weighting was omitted). GB's lower recall (0.52 vs LR's 0.78) partly reflects this difference in imbalance handling; the relative ranking would likely be preserved under equal treatment, but the recall gap would narrow.

* **Experiment Tracking:** Integrated MLflow (`sqlite:///{data.BASE_DIR}/mlflow.db`) to systematically log parameters, metrics, classification reports, confusion matrices, and version-controlled model artifacts.

## 4. Results & Model Selection Analysis

Accuracy is explicitly rejected as a primary evaluation metric—a naive baseline that always predicts "no churn" achieves 73.5% accuracy with 0% recall. Instead, models are judged on **ROC-AUC** and a business-aligned **Recall tiebreaker**.

### Model Performance Comparison

| Model | CV AUC | Test AUC | Precision | Recall | F1 | Verdict | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **`logreg_balanced`** | **0.8460 ± 0.012** | **0.8415** | **0.5043** | **0.7834** | **0.6136** | **Best Recall + F1 (Business Winner)** | 
| `random_forest` | 0.8224 ± 0.011 | 0.8198 | 0.5486 | 0.6337 | 0.5881 | Middle | 
| `gradient_boost` | 0.8483 ± 0.013 | 0.8433 | 0.6655 | 0.5160 | 0.5813 | Best AUC + Precision | 

### Why Logistic Regression is the Business Winner

While Gradient Boosting edges out Logistic Regression on raw Test ROC-AUC (0.8433 vs 0.8415), **the 0.0018 difference is statistically meaningless**—it is roughly 1/7th (14.5%) of the cross-validation standard deviation.

However, their **Recall and Precision profiles diverge dramatically**:

* **Gradient Boosting** catches 51.6% of churners (\~193 True Positives out of 374 actual churners in the test set).

* **Logistic Regression (`logreg_balanced`)** catches 78.3% of churners (\~293 True Positives out of 374 actual churners).

#### Financial & Business Impact Analysis:

* **Additional Churners Caught:** LR identifies **\~100 more churners** in the test set than GB (293 vs. 193). At an assumed Customer Lifetime Value (LTV) of **\$500**, this yields **\~\$50,000 in additional retained revenue**.

* **Cost of False Alarms:** LR trades away precision (0.5043 vs 0.6655), resulting in 288 false positives compared to GB's \~97 false positives (**191 additional false retention offers**). At **\$50 per retention offer**, this adds **\$9,550 in extra marketing spend** (\$50 × 191).

* **Net Expected Value:** Logistic Regression wins by **\~\$40,450 in net expected value** (\$50,000 retained revenue - $9,550 marketing spend), directly satisfying our core business mandate: *missing a churner costs more than a wasted retention offer*.

## 5. How to Run

This project uses `uv` for lightning-fast dependency management and environment isolation.

1. **Install Dependencies:**
   ```bash
   uv sync
   ```

2. **Execute Training Script (Logs to MLflow DB):**
   ```bash
   uv run python -m src.train
   ```

3. **Launch Local MLflow Dashboard:**
   Run from the project root; substitute `$(pwd)` (macOS/Linux) or `%cd%` (Windows) for the absolute path so you don't have to hardcode machine paths:
   ```bash
   uv run mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db"
   ```
   *(Note: MLflow's tracking URI is also derived dynamically from `data.BASE_DIR` in `data.py`, ensuring portability across clone locations.)*

> **Model Artifact Format Note:** MLflow 3.x serializes scikit-learn pipelines using `.skops` (safe serialization rather than legacy pickle). Loading requires `mlflow ≥ 3.0` (or `skops_trusted_types` configuration if loading tree-based artifacts). To load a model programmatically: `mlflow.sklearn.load_model("runs:/<run_id>/model")`.

## 6. Live Demo (TBD)

*Pending deployment of the model serving endpoint via FastAPI.*

## 7. Limitations

* **Precision Tradeoff:** Prioritizing high recall results in a \~50% precision rate, generating false alarms that increase marketing overhead for non-churning customers who receive unnecessary retention incentives.

* **Linear Boundary Constraint:** Logistic Regression relies on linear decision boundaries; highly non-linear feature interactions may require future exploration with tuned tree ensembles once business constraints allow for lower recall thresholds.