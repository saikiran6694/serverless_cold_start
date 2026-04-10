import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config.config import BATCH_SIZE, DEVICE, LSTM_EPOCHS, PATIENCE


DEVICE = torch.device('cuda' if torch.cuda.is_available() else DEVICE)

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM cold-start predictor.

    Architecture:
        Input  (batch, seq_len, n_features)
        → BiLSTM(hidden=64, layers=1)  — smaller to reduce overfitting
        → last hidden state [fwd || bwd] = 128 dims
        → BatchNorm1d(128)
        → Linear(128→32) → ReLU → Dropout(0.3)
        → Linear(32→1) → Sigmoid
    """
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,   # no inter-layer dropout with single layer
        )
        self.bn   = nn.BatchNorm1d(hidden_size * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)
        last    = out[:, -1, :]      # last timestep
        last    = self.bn(last)
        return self.head(last).squeeze(1)


def train_lstm(X_train: np.ndarray, y_train: np.ndarray,
               X_val:   np.ndarray, y_val:   np.ndarray):
    """
    Train BiLSTM with early stopping on validation AUC.
    """
    model     = BiLSTMClassifier(X_train.shape[2]).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # patience=3: drop LR quickly when val plateaus; factor=0.5: halve it
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5,
                                  min_lr=1e-5)
    criterion = nn.BCELoss()

    print(f"\nBiLSTM parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training on: {DEVICE}")

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=BATCH_SIZE)

    history       = {'train_loss': [], 'val_auc': []}
    best_val_auc  = 0.0
    best_weights  = None
    patience_left = PATIENCE

    for epoch in range(1, LSTM_EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(Xb)
        avg_loss = total_loss / len(y_train)

        # Validate
        model.eval()
        val_probs = []
        with torch.no_grad():
            for Xb, _ in val_dl:
                val_probs.append(model(Xb.to(DEVICE)).cpu().numpy())
        val_probs = np.concatenate(val_probs)
        val_auc   = roc_auc_score(y_val, val_probs)

        scheduler.step(val_auc)
        history['train_loss'].append(avg_loss)
        history['val_auc'].append(val_auc)

        # Smooth AUC over last 3 epochs to reduce noise-driven early stopping
        smooth_auc = np.mean(history['val_auc'][-3:])

        print(f"Epoch {epoch:3d}/{LSTM_EPOCHS}  "
              f"loss={avg_loss:.4f}  val_AUC={val_auc:.4f}  "
              f"smooth={smooth_auc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # Early stopping uses smoothed AUC to avoid stopping on a single noisy dip
        if smooth_auc > best_val_auc:
            best_val_auc  = smooth_auc
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch} "
                      f"(best smoothed val_AUC={best_val_auc:.4f})")
                break

    model.load_state_dict(best_weights)
    return model, history


def predict_lstm(model: BiLSTMClassifier, X: np.ndarray) -> np.ndarray:
    model.eval()
    dl = DataLoader(TensorDataset(torch.tensor(X)), batch_size=BATCH_SIZE)
    probs = []
    with torch.no_grad():
        for (Xb,) in dl:
            probs.append(model(Xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(probs)