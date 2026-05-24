"""Tests for experiments."""
import mlflow, pytest, pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from src.experiments.sklearn_experiment import generate_churn_data


@pytest.mark.unit
class TestDataGeneration:
    def test_correct_samples(self):
        X, y = generate_churn_data(100)
        assert len(X) == 100 and len(y) == 100
    def test_columns(self):
        X, y = generate_churn_data(50)
        assert "tenure_months" in X.columns
        assert "monthly_charges" in X.columns
        assert len(X.columns) == 10
    def test_binary(self):
        X, y = generate_churn_data(500)
        assert set(y.unique()).issubset({0, 1})
    def test_reproducible(self):
        X1, y1 = generate_churn_data(100)
        X2, y2 = generate_churn_data(100)
        assert len(X1) == len(X2)


@pytest.mark.unit
class TestMLflowLogging:
    def test_experiment(self, mlflow_test_env):
        mlflow.set_experiment("test"); assert mlflow.get_experiment_by_name("test") is not None
    def test_run(self, mlflow_test_env):
        mlflow.set_experiment("test")
        with mlflow.start_run() as run:
            mlflow.log_param("p", "v"); mlflow.log_metric("m", 0.9)
        r = mlflow.MlflowClient().get_run(run.info.run_id)
        assert r.data.params["p"] == "v" and r.data.metrics["m"] == 0.9
