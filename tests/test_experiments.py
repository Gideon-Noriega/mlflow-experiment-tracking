"""Tests for experiments."""
import mlflow, pytest, pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from src.experiments.sklearn_experiment import create_preprocessing_pipeline, generate_churn_data


@pytest.mark.unit
class TestDataGeneration:
    def test_correct_samples(self): assert len(generate_churn_data(100)) == 100
    def test_columns(self):
        expected = {"tenure_months","monthly_charges","total_charges","num_support_tickets",
            "days_since_last_login","num_logins_last_30d","contract_value","num_products",
            "contract_type","payment_method","subscription_tier","onboarding_completed","churned"}
        assert set(generate_churn_data(50).columns) == expected
    def test_binary(self): assert set(generate_churn_data(500)["churned"].unique()).issubset({0,1})
    def test_reproducible(self):
        pd.testing.assert_frame_equal(generate_churn_data(100,42), generate_churn_data(100,42))


@pytest.mark.unit
class TestPreprocessing:
    def test_transform(self, sample_features, sample_target):
        X = create_preprocessing_pipeline().fit_transform(sample_features)
        assert X.shape[0] == len(sample_features) and not np.isnan(X).any()
    def test_pipeline(self, sample_features, sample_target):
        pipe = Pipeline([("pre", create_preprocessing_pipeline()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42))])
        pipe.fit(sample_features, sample_target)
        assert set(pipe.predict(sample_features)).issubset({0,1})


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
