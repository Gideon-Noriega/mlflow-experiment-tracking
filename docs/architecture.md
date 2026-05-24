# Architecture

```
Data -> [Sklearn | XGBoost | PyTorch] -> MLflow Tracking (PostgreSQL)
                                             |
                                       Artifact Store (MinIO)
                                             |
                                       Model Registry
                                             |
                             Validation Gates + Optuna Tuning
                                             |
                             Promotion (champion/challenger)
                                             |
                             FastAPI Gateway (A/B routing)
```

## Components

| Component | Purpose | Port |
|-----------|---------|------|
| PostgreSQL | MLflow backend store | 5432 |
| MinIO | Artifact storage (S3-compatible) | 9000/9001 |
| MLflow Server | Tracking UI and API | 5000 |
| FastAPI Gateway | Model serving + A/B | 8000 |

## Model Aliases
- `@champion` - production model (90% traffic)
- `@challenger` - evaluation model (10% traffic)
- `@archived` - previous champion for rollback
