"""Experiment Exploration (convert: jupytext --to notebook experiment_exploration.py)"""
# %%
import os, mlflow
from dotenv import load_dotenv
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# %% List experiments
for exp in mlflow.search_experiments():
    print(f"  {exp.name}")

# %% Search runs
exp = mlflow.get_experiment_by_name("customer-churn")
if exp:
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id],
                              order_by=["metrics.test_f1_score DESC"])
    print(runs[["tags.mlflow.runName", "metrics.test_f1_score"]].head(10))

# %% Model registry
client = mlflow.MlflowClient()
try:
    for v in client.search_model_versions("name='customer-churn-model'"):
        print(f"  v{v.version}: {v.run_id[:8]}...")
except Exception as e: print(e)
