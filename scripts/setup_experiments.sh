#!/bin/bash
set -euo pipefail
echo Setup MLflow
pip install -e .[dev]
[ ! -f .env ] && cp .env.example .env
docker compose up -d
sleep 10
python scripts/setup_minio.py
python -m src.experiments.sklearn_experiment
python -m src.experiments.xgboost_experiment
echo Done