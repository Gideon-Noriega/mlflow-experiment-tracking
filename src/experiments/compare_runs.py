"""Compare experiment runs."""
import logging, os, tempfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow, pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_experiment_runs(name):
    exp = mlflow.get_experiment_by_name(name)
    if not exp: raise ValueError(f"Not found: {name}")
    return mlflow.search_runs(experiment_ids=[exp.experiment_id], order_by=["metrics.test_f1_score DESC"])


def compare_metrics(runs_df):
    valid = runs_df[runs_df["status"] == "FINISHED"]
    if valid.empty: return
    names = valid["tags.mlflow.runName"].fillna("unnamed")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, m, t in zip(axes, ["metrics.test_accuracy", "metrics.test_f1_score", "metrics.test_auc_roc"],
                        ["Accuracy", "F1", "AUC"]):
        if m in valid.columns:
            ax.barh(range(len(names)), valid[m].values); ax.set_title(t)
    p = tempfile.mktemp(suffix=".png"); fig.savefig(p); plt.close()
    with mlflow.start_run(run_name="run-comparison"):
        mlflow.log_artifact(p, "comparison")


def find_best_run(name, metric="metrics.test_f1_score"):
    runs = get_experiment_runs(name)
    valid = runs[runs["status"] == "FINISHED"]
    return valid.loc[valid[metric].idxmax()]["run_id"]


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    name = os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn")
    mlflow.set_experiment(name)
    runs = get_experiment_runs(name)
    logger.info(f"Found {len(runs)} runs")
    compare_metrics(runs)
    logger.info(f"Best: {find_best_run(name)}")


if __name__ == "__main__":
    main()
