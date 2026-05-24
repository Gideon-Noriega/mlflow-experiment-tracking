# MLflow Experiment Tracking

A production-grade MLflow showcase demonstrating the full ML lifecycle: experiment tracking, hyperparameter tuning, model registry, validation gates, and model serving with A/B testing.

**Use Case:** Customer churn prediction for a SaaS company with multiple model experiments, automated tuning, and production deployment patterns.

## Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Data Pipeline   |     |   MLflow Server   |     |   Model Serving   |
|                   |     |                   |     |                   |
|  Feature Eng. ----+---->|  Tracking Server  |     |  FastAPI Gateway  |
|  Train / Tune     |     |  (PostgreSQL)     |<----+  A/B Router       |
|  Validate         |     |                   |     |  Health Checks    |
+-------------------+     +--------+----------+     +-------------------+
                                   |
                          +--------+----------+
                          |  Artifact Store   |
                          |  (MinIO / S3)     |
                          |                   |
                          |  - Models         |
                          |  - Plots          |
                          |  - Data Profiles  |
                          +-------------------+

+---------------------------------------------------------------------------+
|                         Model Registry                                     |
|                                                                           |
|  [Experiment] --> [Run] --> [Register] --> [Validate] --> [Promote]       |
|                                                                           |
|  Aliases:  @champion  @challenger  @archived                              |
|  Stages:   None -> Staging -> Production -> Archived                      |
+---------------------------------------------------------------------------+

+---------------------------------------------------------------------------+
|                      Hyperparameter Tuning                                 |
|                                                                           |
|  Optuna Study  ------>  MLflow Tracking  ------>  Best Model Registration |
|  (TPE Sampler)          (params/metrics)          (auto-promotion)        |
+---------------------------------------------------------------------------+
```

## Features

- **Experiment Tracking**: Parameters, metrics, artifacts, and tags with full lineage
- **Autologging**: Sklearn, XGBoost, and PyTorch with automatic metric capture
- **Model Registry**: Full lifecycle management with aliases (champion/challenger)
- **Model Signatures**: Input/output schema validation with examples
- **Custom PyFunc Models**: Ensemble model flavor with preprocessing pipeline
- **Model Serving**: FastAPI-based REST API with A/B traffic routing
- **Hyperparameter Tuning**: Optuna integration with MLflow callback
- **Validation Gates**: Automated quality checks before model promotion
- **Artifact Storage**: MinIO (S3-compatible) backend for model artifacts
- **Tracking Server**: PostgreSQL-backed MLflow server with Docker Compose
- **Run Comparison**: Programmatic experiment analysis and visualization
- **CI/CD**: GitHub Actions workflow for training, validation, and deployment

## Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Make

### Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/mlflow-experiment-tracking.git
cd mlflow-experiment-tracking

# Start infrastructure (MLflow server, MinIO, PostgreSQL)
make infra-up

# Install Python dependencies
make install

# Run the full experiment pipeline
make train

# Launch MLflow UI
make ui

# Serve the champion model
make serve

# Run A/B testing gateway
make gateway
```

### One-Command Demo

```bash
make demo  # Runs: infra-up -> train -> tune -> promote -> serve
```

## Project Structure

```
mlflow-experiment-tracking/
|-- src/
|   |-- experiments/         # Training scripts per framework
|   |   |-- sklearn_experiment.py
|   |   |-- xgboost_experiment.py
|   |   |-- pytorch_experiment.py
|   |   |-- compare_runs.py
|   |-- models/              # Custom model flavors
|   |   |-- ensemble_model.py
|   |   |-- preprocessing_pipeline.py
|   |-- registry/            # Model lifecycle management
|   |   |-- promote.py
|   |   |-- validation_gates.py
|   |-- serving/             # Model serving
|   |   |-- app.py
|   |   |-- ab_router.py
|   |-- tuning/              # Hyperparameter optimization
|       |-- optuna_tuning.py
|-- tests/
|-- configs/
|-- scripts/
|-- notebooks/
|-- docs/
|-- docker-compose.yml
|-- pyproject.toml
|-- Makefile
```

## Model Lifecycle

```
Train -> Register -> Validate -> Stage -> Promote -> Serve -> Monitor -> Retrain
                        |                    |
                        v                    v
                   [REJECT]           [A/B Test]
                   (alert)            (challenger)
```

See [docs/model-lifecycle.md](docs/model-lifecycle.md) for details.

## Configuration

Copy `.env.example` to `.env` and adjust:

```bash
cp .env.example .env
```

Key settings:
- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://localhost:5000)
- `MLFLOW_S3_ENDPOINT_URL`: MinIO endpoint (default: http://localhost:9000)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`: MinIO credentials

## License

MIT
