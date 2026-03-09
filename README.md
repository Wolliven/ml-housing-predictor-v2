# ML Housing Price Predictor

A professional-grade classical machine learning project focused on building a robust, reproducible regression pipeline for housing price prediction.

This version upgrades the original implementation by introducing:

* Preprocessing pipelines
* Feature scaling
* Regularized regression
* Cross-validation
* Model comparison
* Self-describing model artifacts

The goal is depth, not feature expansion.

---

## Project Objectives

This project aims to implement a production-style ML training workflow that:

1. Loads structured housing data
2. Applies proper preprocessing inside a pipeline
3. Compares multiple regression models
4. Evaluates performance using cross-validation
5. Selects the best-performing model programmatically
6. Saves a fully self-contained model artifact
7. Performs inference with strict feature validation

This project emphasizes correctness, evaluation discipline, and reproducibility.

---

## Key Improvements Over v1

* Preprocessing handled inside `sklearn.pipeline.Pipeline`
* Feature scaling using `StandardScaler`
* Regularized regression via `Ridge`
* K-fold cross-validation for robust evaluation
* Automatic best model selection
* Metadata-rich model persistence
* Explicit feature contract enforcement

---

## Iterative Model Development (v3)

After establishing a stable training pipeline in v2, the project continued with a third development phase focused on improving model understanding and performance.

This phase introduced a structured experimentation workflow including:

* Feature engineering of derived housing indicators
* Error analysis of model predictions
* Inspection of extreme prediction failures
* Model coefficient interpretation for explainability

New engineered features include:

* `households = Population / AveOccup`
* `people_per_bedroom = AveOccup / AveBedrms`
* `bedrooms_per_room = AveBedrms / AveRooms`

These experiments demonstrated that Ridge regression benefits significantly from the engineered features due to its ability to handle correlated predictors through regularization.

All experiments and analyses are documented in the `reports/` directory.

---

## Project Structure

```
ml-housing-predictor-v2/
├── data/
├── input/
├── reports/
│   ├── baseline_analysis.md
│   ├── feature_engineering_v1.md
│   └── model_interpretation_v1.md
├── scripts/
│   └── analyze_models.py
├── train.py
├── predict.py
├── ml_engine.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Training Workflow

Training performs the following steps:

* Load dataset
* Split features and target
* Build preprocessing + model pipeline
* Evaluate candidate models using cross-validation
* Compare mean and standard deviation of scores
* Select best model
* Train final model
* Save artifact containing:

  * trained pipeline
  * feature schema
  * cross-validation summary for compared models
  * selected model information
  * dataset and model file metadata
---

## Model Comparison

This version compares at least:

* LinearRegression
* Ridge Regression (L2 regularization)

Models are evaluated using K-fold cross-validation.

Selection is automatic based on the chosen scoring metric.

---

## Usage

### Train a model

```
python train.py data.csv [--model_path model.pkl]
```

This will:

* Evaluate candidate models
* Print comparison results
* Train the selected model
* Save a structured model artifact

---

### Make predictions

```
python predict.py [--model model.pkl] input.json [--out output.json/csv]
```

This will:

* Load the saved artifact
* Validate input features
* Apply preprocessing
* Generate predictions

---

## Evaluation Strategy

Evaluation is performed using:

* Cross-validation (K-fold)
* Mean score
* Standard deviation

This ensures stable performance estimation and reduces dependency on a single train/test split.

---

## Scope

This project intentionally focuses on classical machine learning foundations.

Out of scope:

* Deep learning
* Hyperparameter search frameworks
* Deployment
* APIs
* Distributed systems
* Model serving infrastructure

---

## Learning Focus

This project strengthens understanding of:

* Data leakage prevention
* Pipeline architecture
* Regularization and bias-variance tradeoff
* Proper model evaluation
* Reproducible ML artifacts
* Engineering-oriented ML design
* Feature engineering and model improvement
* Model interpretability through coefficient analysis

---

## Requirements

* Python 3.9+
* pandas
* numpy
* scikit-learn

Install dependencies:

```
pip install -r requirements.txt
```

---

## Philosophy

This project prioritizes:

* Robustness over shortcuts
* Clarity over complexity
* Reproducibility over experimentation
* Engineering discipline over notebook-style workflows