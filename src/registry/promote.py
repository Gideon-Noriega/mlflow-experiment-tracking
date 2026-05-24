"""Model promotion script."""
import logging, os
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from src.registry.validation_gates import validate_model_quality

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_best_model_version(client, model_name, metric="test_f1_score"):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions: return None
    best_v, best_val = None, -1.0
    for v in versions:
        try:
            val = client.get_run(v.run_id).data.metrics.get(metric, 0)
            if val > best_val: best_val, best_v = val, int(v.version)
        except Exception: pass
    logger.info(f"Best: v{best_v} ({metric}={best_val:.4f})")
    return best_v


def promote_to_champion(client, model_name, version, skip_validation=False):
    logger.info(f"Promoting {model_name} v{version}")
    if not skip_validation:
        uri = f"models:/{model_name}/{version}"
        run = client.get_run(client.get_model_version(model_name, version).run_id)
        ok, _ = validate_model_quality(uri, run.data.metrics)
        if not ok:
            logger.error("Validation failed")
            client.set_model_version_tag(model_name, version, "validation", "failed")
            return False
    try:
        old = client.get_model_version_by_alias(model_name, "champion")
        client.set_registered_model_alias(model_name, "archived", old.version)
        logger.info(f"Archived v{old.version}")
    except Exception: pass
    client.set_registered_model_alias(model_name, "champion", str(version))
    logger.info(f"Promoted v{version} to champion!")
    return True


def set_challenger(client, model_name, version):
    client.set_registered_model_alias(model_name, "challenger", str(version))
    logger.info(f"Set v{version} as challenger")


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mn = os.getenv("MODEL_NAME", "customer-churn-model")
    client = MlflowClient()
    best = get_best_model_version(client, mn)
    if best: promote_to_champion(client, mn, best)


if __name__ == "__main__":
    main()
