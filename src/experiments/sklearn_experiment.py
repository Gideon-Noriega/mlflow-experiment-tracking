"""Scikit-learn experiment with autologging and manual metric tracking."""
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def generate_churn_data(n_samples: int = 10000) -> tuple[pd.DataFrame, pd.Series]:
    np.random.seed(42)
    data = pd.DataFrame({
        "tenure_months": np.random.randint(1, 72, n_samples),
        "monthly_charges": np.random.uniform(20, 100, n_samples),
        "total_charges": np.random.uniform(100, 7000, n_samples),
        "num_support_tickets": np.random.poisson(2, n_samples),
        "contract_type": np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
        "payment_method": np.random.choice([0, 1, 2, 3], n_samples),
        "num_products": np.random.randint(1, 6, n_samples),
        "avg_session_duration": np.random.exponential(15, n_samples),
        "login_frequency": np.random.poisson(10, n_samples),
        "days_since_last_login": np.random.exponential(7, n_samples),
    })

    churn_prob = 1 / (1 + np.exp(-(
        -2 + 0.03 * data["num_support_tickets"]
        - 0.02 * data["tenure_months"]
        + 0.01 * data["monthly_charges"]
        - 0.5 * data["contract_type"]
        + 0.1 * data["days_since_last_login"]
    )))
    target = (np.random.random(n_samples) < churn_prob).astype(int)

    return data, target


def run_experiment():
    mlflow.set_experiment("churn-prediction-sklearn")

    X, y = generate_churn_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, C=0.1)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42,
        ),
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            mlflow.sklearn.autolog(log_models=True)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

            mlflow.log_metrics({
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_precision": precision_score(y_test, y_pred),
                "test_recall": recall_score(y_test, y_pred),
                "test_f1": f1_score(y_test, y_pred),
                "test_auc_roc": roc_auc_score(y_test, y_proba),
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
            })

            mlflow.log_params({
                "model_type": model_name,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": list(X.columns),
                "churn_rate": y.mean(),
            })

            mlflow.set_tags({
                "dataset": "synthetic_churn",
                "stage": "experimentation",
                "author": "gideon",
            })

            print(f"{model_name}: F1={f1_score(y_test, y_pred):.4f}, AUC={roc_auc_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    run_experiment()
