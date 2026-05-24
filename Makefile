.PHONY: install train tune promote serve gateway ui infra-up infra-down test lint demo compare clean

PYTHON := python
DOCKER_COMPOSE := docker compose

# Setup
install:
	pip install -e ".[dev,pytorch]"
	pre-commit install || true

# Infrastructure
infra-up:
	$(DOCKER_COMPOSE) up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@$(PYTHON) scripts/setup_minio.py
	@echo "MLflow UI: http://localhost:5000"
	@echo "MinIO Console: http://localhost:9001"

infra-down:
	$(DOCKER_COMPOSE) down -v

# Training
train: train-sklearn train-xgboost
	@echo "All experiments complete. View at http://localhost:5000"

train-sklearn:
	$(PYTHON) -m src.experiments.sklearn_experiment

train-xgboost:
	$(PYTHON) -m src.experiments.xgboost_experiment

train-pytorch:
	$(PYTHON) -m src.experiments.pytorch_experiment

# Hyperparameter Tuning
tune:
	$(PYTHON) -m src.tuning.optuna_tuning

# Model Registry
promote:
	$(PYTHON) -m src.registry.promote

validate:
	$(PYTHON) -m src.registry.validation_gates

# Model Serving
serve:
	mlflow models serve -m "models:/customer-churn-model@champion" --port 8080 --no-conda

gateway:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

# Analysis
compare:
	$(PYTHON) -m src.experiments.compare_runs

ui:
	mlflow ui --host 0.0.0.0 --port 5000

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/ -v -m unit

test-integration:
	pytest tests/ -v -m integration

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

# Demo (full pipeline)
demo: infra-up train tune promote
	@echo ""
	@echo "======================================="
	@echo "  Demo complete!"
	@echo "  MLflow UI: http://localhost:5000"
	@echo "  Run 'make gateway' to start A/B serving"
	@echo "======================================="

# Cleanup
clean:
	rm -rf mlruns/ mlartifacts/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
