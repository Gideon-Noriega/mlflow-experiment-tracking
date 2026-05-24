"""Custom PyFunc ensemble model."""
import json, logging, tempfile
import mlflow, mlflow.pyfunc
import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

logger = logging.getLogger(__name__)


class ChurnEnsembleModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_uris=None, weights=None, strategy="weighted_average"):
        self.model_uris = model_uris or []
        self.weights = weights
        self.strategy = strategy
        self.models = []

    def load_context(self, context):
        cfg = context.artifacts.get("ensemble_config")
        if cfg:
            with open(cfg) as f: c = json.load(f)
            self.model_uris = c["model_uris"]
            self.weights = c.get("weights")
        self.models = [mlflow.pyfunc.load_model(u) for u in self.model_uris]
        if not self.weights: self.weights = [1.0/len(self.models)]*len(self.models)

    def predict(self, context, model_input, params=None):
        preds = np.array([m.predict(model_input) for m in self.models])
        w = np.array(self.weights).reshape(-1, 1)
        prob = (preds * w).sum(axis=0)
        return (prob > 0.5).astype(int)


def get_ensemble_signature():
    inputs = Schema([ColSpec("double", "tenure_months"), ColSpec("double", "monthly_charges"),
        ColSpec("double", "total_charges"), ColSpec("double", "num_support_tickets"),
        ColSpec("double", "days_since_last_login"), ColSpec("double", "num_logins_last_30d"),
        ColSpec("double", "contract_value"), ColSpec("double", "num_products"),
        ColSpec("string", "contract_type"), ColSpec("string", "payment_method"),
        ColSpec("string", "subscription_tier"), ColSpec("string", "onboarding_completed")])
    return ModelSignature(inputs=inputs, outputs=Schema([ColSpec("integer", "prediction")]))


def register_ensemble(model_uris, weights=None, name="customer-churn-ensemble"):
    cfg = {"model_uris": model_uris, "weights": weights}
    p = tempfile.mktemp(suffix=".json")
    with open(p, "w") as f: json.dump(cfg, f)
    with mlflow.start_run(run_name="ensemble"):
        return mlflow.pyfunc.log_model("model", python_model=ChurnEnsembleModel(model_uris, weights),
            artifacts={"ensemble_config": p}, signature=get_ensemble_signature(),
            registered_model_name=name).model_uri
