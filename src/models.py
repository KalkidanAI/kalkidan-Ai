"""
models.py

Model wrappers with common interface for Agricultural Drought Early Warning:
- RuleBaseline: Agronomic threshold model (VCI-based)
- DroughtRF: Random Forest classifier with class balancing
- DroughtXGB: Gradient-boosted decision trees with early stopping
- DroughtLSTM: PyTorch recurrent neural network for temporal sequence forecasting
"""

import os
import json
import numpy as np
import pandas as pd

# Optional dependencies with safe imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
    import joblib
except ImportError:
    RandomForestClassifier = None
    joblib = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None
    nn = None


def compute_metrics(y_true, y_pred, positive_classes=(1, 2, 3)):
    """Computes F1-macro, Accuracy, FAR (False Alarm Ratio), and POD (Probability of Detection)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    acc = float(np.mean(y_true == y_pred))
    
    # Drought positive is class > 0 (Watch, Warning, Severe)
    is_drought_true = np.isin(y_true, positive_classes)
    is_drought_pred = np.isin(y_pred, positive_classes)
    
    tp = int(np.sum(is_drought_true & is_drought_pred))
    fp = int(np.sum(~is_drought_true & is_drought_pred))
    fn = int(np.sum(is_drought_true & ~is_drought_pred))
    
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    pod = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Calculate macro F1
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        c_tp = np.sum((y_true == c) & (y_pred == c))
        c_fp = np.sum((y_true != c) & (y_pred == c))
        c_fn = np.sum((y_true == c) & (y_pred != c))
        denom = (2 * c_tp + c_fp + c_fn)
        f1_c = (2 * c_tp / denom) if denom > 0 else 0.0
        f1s.append(f1_c)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    
    return {
        "Accuracy": round(acc, 4),
        "Macro_F1": round(macro_f1, 4),
        "FAR": round(far, 4),
        "POD": round(pod, 4),
        "TP": tp,
        "FP": fp,
        "FN": fn
    }


class RuleBaseline:
    """Operational agronomic rule-based baseline model using VCI thresholds."""
    def __init__(self, vci_col="vci"):
        self.vci_col = vci_col

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        """
        Classifies based on VCI thresholds:
        0: Normal (> 40)
        1: Watch (35 - 40)
        2: Warning (20 - 35)
        3: Severe (<= 20)
        """
        if isinstance(X, pd.DataFrame) and self.vci_col in X.columns:
            vci = X[self.vci_col].values
        elif isinstance(X, np.ndarray):
            vci = X[:, 0]  # assumes first column is VCI
        else:
            raise ValueError(f"X must be a DataFrame containing '{self.vci_col}' or a 2D numpy array.")
            
        preds = np.zeros(len(vci), dtype=int)
        preds[(vci <= 40) & (vci > 35)] = 1
        preds[(vci <= 35) & (vci > 20)] = 2
        preds[vci <= 20] = 3
        return preds

    def evaluate(self, X, y):
        preds = self.predict(X)
        return compute_metrics(y, preds)


class DroughtRF:
    """Random Forest Classifier for multimodal drought early warning."""
    def __init__(self, n_estimators=300, max_depth=15, min_samples_leaf=20,
                 class_weight="balanced", random_state=42, n_jobs=-1):
        if RandomForestClassifier is None:
            raise ImportError("scikit-learn is required to use DroughtRF.")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_names = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return compute_metrics(y, preds)

    def feature_importance(self, feature_names=None):
        names = feature_names or self.feature_names
        importances = self.model.feature_importances_
        if names:
            return pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        return importances

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names = data.get("feature_names")
        return instance


class DroughtXGB:
    """XGBoost Classifier with support for early stopping and ablation."""
    def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
                 early_stopping_rounds=30, random_state=42):
        if xgb is None:
            raise ImportError("xgboost is required to use DroughtXGB.")
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "eval_metric": eval_metric,
            "random_state": random_state,
            "early_stopping_rounds": early_stopping_rounds
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names = None

    def fit(self, X, y, eval_set=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
            
        formatted_eval = None
        if eval_set is not None:
            formatted_eval = []
            for (eval_x, eval_y) in eval_set:
                if isinstance(eval_x, pd.DataFrame): eval_x = eval_x.values
                if isinstance(eval_y, (pd.Series, pd.DataFrame)): eval_y = eval_y.values.ravel()
                formatted_eval.append((eval_x, eval_y))

        self.model.fit(X, y, eval_set=formatted_eval, verbose=False)
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return compute_metrics(y, preds)

    def feature_importance(self, feature_names=None):
        names = feature_names or self.feature_names
        importances = self.model.feature_importances_
        if names:
            return pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
        return importances

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        if filepath.endswith(".json"):
            self.model.save_model(filepath)
            # Save feature names alongside
            meta_path = filepath + ".meta"
            with open(meta_path, "w") as f:
                json.dump({"feature_names": self.feature_names}, f)
        else:
            joblib.dump({"model": self.model, "feature_names": self.feature_names}, filepath)

    @classmethod
    def load(cls, filepath):
        instance = cls()
        if filepath.endswith(".json"):
            instance.model.load_model(filepath)
            meta_path = filepath + ".meta"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    instance.feature_names = json.load(f).get("feature_names")
        else:
            data = joblib.load(filepath)
            instance.model = data["model"]
            instance.feature_names = data.get("feature_names")
        return instance


if torch is not None:
    class _PyTorchLSTMNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=1, num_classes=4, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]
            dropped = self.dropout(last_step)
            return self.fc(dropped)

    class _PyTorchDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
else:
    _PyTorchLSTMNet = None
    _PyTorchDataset = None


class DroughtLSTM:
    """PyTorch LSTM Sequence Model for temporal drought forecasting."""
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=1, num_classes=4,
                 dropout=0.3, lr=0.001, batch_size=256, epochs=30, patience=7, device=None):
        if torch is None:
            raise ImportError("PyTorch is required to use DroughtLSTM.")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.net = _PyTorchLSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout
        ).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fits LSTM using sliding sequence batches with early stopping."""
        train_ds = _PyTorchDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = _PyTorchDataset(X_val, y_val)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self.net.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
            train_loss /= len(train_ds)

            # Validation
            if val_loader is not None:
                self.net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.net(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * len(batch_x)
                val_loss /= len(val_ds)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.to(self.device)
            
        return self

    def predict_proba(self, X):
        self.net.eval()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.net(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return compute_metrics(y, preds)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "dropout": self.dropout
            }
        }, filepath)

    @classmethod
    def load(cls, filepath, device=None):
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        cfg = checkpoint["config"]
        instance = cls(**cfg, device=device)
        instance.net.load_state_dict(checkpoint["state_dict"])
        return instance


def temporal_train_val_test_split(df: pd.DataFrame, train_end: str = "2022-01-01",
                                  val_end: str = "2023-07-01", date_col: str = "date"):
    """Splits tabular time series data strictly by date thresholds to avoid temporal data leakage."""
    dates = pd.to_datetime(df[date_col])
    train = df[dates <= pd.to_datetime(train_end)].copy()
    val = df[(dates > pd.to_datetime(train_end)) & (dates <= pd.to_datetime(val_end))].copy()
    test = df[dates > pd.to_datetime(val_end)].copy()
    return train, val, test


def create_sequences(df: pd.DataFrame, feature_cols: list, target_col: str,
                     seq_len: int = 8, group_col: str = "cell_id"):
    """
    Constructs sliding window sequences for temporal neural networks.
    Each sequence consists of seq_len timesteps of features predicting the target at the final timestep.
    """
    X_list, y_list = [], []
    for _, group in df.groupby(group_col):
        group_sorted = group.sort_values("date") if "date" in group.columns else group
        feats = group_sorted[feature_cols].values
        targets = group_sorted[target_col].values
        n_samples = len(feats)
        for i in range(n_samples - seq_len + 1):
            X_list.append(feats[i:i + seq_len])
            y_list.append(targets[i + seq_len - 1])
            
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols))), np.empty((0,), dtype=int)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=int)
