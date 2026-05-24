# Model Lifecycle

```
Train -> Register -> Validate -> Challenger -> Champion -> Archived
                        |
                    [FAIL] -> Alert
```

## Validation Gates

| Gate | Threshold |
|------|-----------|
| Accuracy | >= 0.80 |
| F1 Score | >= 0.75 |
| AUC-ROC | >= 0.82 |
| Latency | < 50ms |
| vs Champion | >= 1% improvement |

## A/B Testing
- Champion: 90% traffic
- Challenger: 10% traffic
- Min 500 observations before promotion

## CLI
```bash
python scripts/promote_model.py --best --alias champion
python scripts/promote_model.py --version 5 --alias challenger
```
