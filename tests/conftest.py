"""Pytest fixtures."""
import os, mlflow
import numpy as np, pandas as pd
import pytest


@pytest.fixture(autouse=True)
def mlflow_test_env(tmp_path):
    uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(uri); os.environ["MLFLOW_TRACKING_URI"] = uri
    yield uri


@pytest.fixture
def sample_churn_data():
    rng = np.random.default_rng(42); n = 100
    return pd.DataFrame({"tenure_months": rng.integers(1,72,n).astype(float),
        "monthly_charges": rng.uniform(30,300,n).round(2),
        "total_charges": rng.uniform(100,20000,n).round(2),
        "num_support_tickets": rng.integers(0,10,n).astype(float),
        "days_since_last_login": rng.integers(0,60,n).astype(float),
        "num_logins_last_30d": rng.integers(0,50,n).astype(float),
        "contract_value": rng.uniform(500,50000,n).round(2),
        "num_products": rng.integers(1,8,n).astype(float),
        "contract_type": rng.choice(["month-to-month","one-year","two-year"],n),
        "payment_method": rng.choice(["credit_card","bank_transfer","invoice"],n),
        "subscription_tier": rng.choice(["basic","professional","enterprise"],n),
        "onboarding_completed": rng.choice(["yes","no"],n),
        "churned": rng.integers(0,2,n)})


@pytest.fixture
def sample_features(sample_churn_data): return sample_churn_data.drop(columns=["churned"])


@pytest.fixture
def sample_target(sample_churn_data): return sample_churn_data["churned"].values
