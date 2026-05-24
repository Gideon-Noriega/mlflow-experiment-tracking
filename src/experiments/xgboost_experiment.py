"""XGBoost experiment with custom metrics and feature importance logging."""
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

from .sklearn_experiment import generate_churn_data


def run_xgboost_experiment():
    mlflow.set_experiment("churn-prediction-xgboost")

    X, y = generate_churn_data(n_samples=20000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X.columns))

    param_sets = [
        {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 200, "subsample": 0.8},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.9},
        {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 500, "subsample": 0.7},
    ]

    for i, params in enumerate(param_sets):
        with mlflow.start_run(run_name=f"xgb_config_{i}"):
            mlflow.xgboost.autolog()

            xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": ["logloss", "auc"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "subsample": params["subsample"],
                "colsample_bytree": 0.8,
                "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
                "tree_method": "hist",
                "random_state": 42,
            }

            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params["n_estimators"],
                evals=[(dtrain, "train"), (dtest, "eval")],
                early_stopping_rounds=20,
                verbose_eval=False,
            )

            y_proba = model.predict(dtest)
            y_pred = (y_proba > 0.5).astype(int)

            mlflow.log_metrics({
                "test_f1": f1_score(y_test, y_pred),
                "test_auc": roc_auc_score(y_test, y_proba),
                "best_iteration": model.best_iteration,
            })

            importance = model.get_score(importance_type="gain")
            mlflow.log_dict(importance, "feature_importance.json")

            print(f"Config {i}: F1={f1_score(y_test, y_pred):.4f}, "
                  f"AUC={roc_auc_score(y_test, y_proba):.4f}, "
                  f"best_iter={model.best_iteration}")


if __name__ == "__main__":
    run_xgboost_experiment()
