#!/usr/bin/env python
"""CLI for model promotion."""
import argparse, logging, os, sys
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.registry.promote import get_best_model_version, promote_to_champion, set_challenger

load_dotenv()
logging.basicConfig(level=logging.INFO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="customer-churn-model")
    p.add_argument("--version", type=int)
    p.add_argument("--best", action="store_true")
    p.add_argument("--alias", choices=["champion","challenger"], default="champion")
    p.add_argument("--skip-validation", action="store_true")
    args = p.parse_args()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    v = get_best_model_version(client, args.model_name) if args.best else args.version
    if not v: sys.exit(1)
    if args.alias == "champion":
        if not promote_to_champion(client, args.model_name, v, args.skip_validation): sys.exit(1)
    else: set_challenger(client, args.model_name, v)


if __name__ == "__main__": main()
