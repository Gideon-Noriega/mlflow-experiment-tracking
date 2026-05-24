"""A/B testing router."""
import logging, random
from collections import defaultdict
import mlflow, mlflow.pyfunc
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class ABRouter:
    def __init__(self, model_name, tracking_uri, champion_weight=0.9, challenger_weight=0.1):
        self.model_name = model_name
        self.tracking_uri = tracking_uri
        self.champion_weight = champion_weight
        self.challenger_weight = challenger_weight
        self.champion_model = self.challenger_model = None
        self.champion_version = self.challenger_version = None
        self.stats = defaultdict(lambda: {"predictions": 0, "total_latency_ms": 0.0, "churn_count": 0})

    def load_models(self):
        client = MlflowClient(self.tracking_uri)
        try:
            info = client.get_model_version_by_alias(self.model_name, "champion")
            self.champion_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@champion")
            self.champion_version = info.version
        except Exception as e: logger.warning(f"No champion: {e}")
        try:
            info = client.get_model_version_by_alias(self.model_name, "challenger")
            self.challenger_model = mlflow.pyfunc.load_model(f"models:/{self.model_name}@challenger")
            self.challenger_version = info.version
        except Exception: pass

    def is_ready(self): return self.champion_model is not None

    def route(self):
        if not self.champion_model: raise RuntimeError("No models")
        if not self.challenger_model or random.random() < self.champion_weight:
            return self.champion_model, "champion", self.champion_version or "?"
        return self.challenger_model, "challenger", self.challenger_version or "?"

    def log_prediction(self, alias, pred, latency):
        self.stats[alias]["predictions"] += 1
        self.stats[alias]["total_latency_ms"] += latency
        if pred == 1: self.stats[alias]["churn_count"] += 1

    def get_stats(self):
        return {a: {"predictions": s["predictions"],
                    "avg_latency": s["total_latency_ms"]/max(s["predictions"],1),
                    "churn_rate": s["churn_count"]/max(s["predictions"],1)}
                for a, s in self.stats.items()}
