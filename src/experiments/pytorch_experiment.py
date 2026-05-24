"""PyTorch experiment for churn prediction."""
import logging, os
import mlflow, mlflow.pytorch
import numpy as np, pandas as pd
import torch, torch.nn as nn
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from src.experiments.sklearn_experiment import generate_churn_data

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.extend([nn.Linear(prev, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "customer-churn"))
    data = generate_churn_data(5000)
    df = data.copy()
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = LabelEncoder().fit_transform(df[c])
    feats = [c for c in df.columns if c != "churned"]
    X = StandardScaler().fit_transform(df[feats].values.astype(np.float32))
    y = df["churned"].values.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_ld = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr)), batch_size=64, shuffle=True)
    test_ld = DataLoader(TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(y_te)), batch_size=64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with mlflow.start_run(run_name="pytorch-churnnet"):
        mlflow.set_tags({"model_type": "neural-network", "framework": "pytorch", "engineer": "gideon"})
        mlflow.log_params({"hidden": "128-64-32", "epochs": 50, "lr": 0.001})
        model = ChurnNet(X_tr.shape[1], [128, 64, 32]).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.BCELoss()
        best_f1 = 0.0
        for ep in range(50):
            model.train(); tl = 0.0
            for bx, by in train_ld:
                bx, by = bx.to(dev), by.to(dev)
                opt.zero_grad(); loss = crit(model(bx).squeeze(), by)
                loss.backward(); opt.step(); tl += loss.item()
            model.eval(); ps, pbs = [], []
            with torch.no_grad():
                for bx, _ in test_ld:
                    o = model(bx.to(dev)).squeeze()
                    pbs.extend(o.cpu().numpy()); ps.extend((o>0.5).int().cpu().numpy())
            ef1 = f1_score(y_te, ps); eauc = roc_auc_score(y_te, pbs)
            mlflow.log_metrics({"train_loss": tl/len(train_ld), "val_f1": ef1, "val_auc": eauc}, step=ep)
            if ef1 > best_f1: best_f1 = ef1
        mlflow.log_metrics({"test_f1_score": ef1, "test_auc_roc": eauc, "best_f1": best_f1})
        model.eval()
        with torch.no_grad():
            so = model(torch.FloatTensor(X_te[:3]).to(dev)).cpu().numpy()
        sig = infer_signature(pd.DataFrame(X_te[:3], columns=feats), so)
        mlflow.pytorch.log_model(model, "model", signature=sig,
            input_example=pd.DataFrame(X_te[:3], columns=feats),
            registered_model_name="customer-churn-model")
        logger.info(f"PyTorch done: best_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
