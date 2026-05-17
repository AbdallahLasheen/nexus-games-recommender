"""
Neural Collaborative Filtering (NCF) — Offline Training Script
AIE425 Intelligent Recommender System — Lecture 7

Architecture (exactly as described in Lecture 7):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Input: (user_idx, item_idx)                                        │
  │  ↓                                                                  │
  │  Embedding Layer  → user_emb (dim=64), item_emb (dim=64)           │
  │  ↓                                                                  │
  │  Concatenate → [user_emb || item_emb]  (dim=128)                   │
  │  ↓                                                                  │
  │  FC(128→64) + ReLU                                                  │
  │  FC(64→32)  + ReLU                                                  │
  │  FC(32→16)  + ReLU                                                  │
  │  FC(16→1)   + Sigmoid   → predicted preference score ∈ (0,1)       │
  └─────────────────────────────────────────────────────────────────────┘

Training:
  - Convert ratings > 3 → positive (1), else → 0   (binary classification)
  - Loss: Binary Cross-Entropy
  - Optimizer: Adam (lr=0.001)
  - Saved to: models/saved/ncf_model.pt
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─── Dataset ─────────────────────────────────────────────────────────────────

class RatingDataset(Dataset):
    def __init__(self, user_ids: np.ndarray, item_ids: np.ndarray, labels: np.ndarray):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.labels   = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


# ─── NCF Model ───────────────────────────────────────────────────────────────

class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering.
    Embedding → Concat → FC(ReLU) × 3 → FC(Sigmoid)
    Exactly as in Lecture 7.
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        super().__init__()

        # Step 1: Embedding Layer (Lecture 7, Step 1)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # Step 2–4: Fully-Connected layers with ReLU, final Sigmoid (Lecture 7)
        self.fc_layers = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # Embedding look-up
        u_emb = self.user_emb(user_ids)   # (batch, emb_dim)
        i_emb = self.item_emb(item_ids)   # (batch, emb_dim)

        # Concatenate (Lecture 7, Step 2)
        x = torch.cat([u_emb, i_emb], dim=1)   # (batch, emb_dim*2)

        # FC stack + Sigmoid
        out = self.fc_layers(x)   # (batch, 1)
        return out.squeeze(1)      # (batch,)


# ─── Training function ────────────────────────────────────────────────────────

def train_ncf(
    df_cf: pd.DataFrame,
    n_users: int,
    n_items: int,
    emb_dim: int = 64,
    lr: float = 0.001,
    n_epochs: int = 20,
    batch_size: int = 1024,
    save_path: str = "models/saved/ncf_model.pt",
    device: str = None,
) -> NCFModel:
    """
    Train the NCF model offline.
    Returns the trained model and saves it to save_path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training NCF on device: {device}")

    # Binary labels: rating > 3 → 1, else 0  (Lecture 5 evaluation convention)
    df = df_cf[["user_idx", "item_idx", "rating"]].copy()
    df["label"] = (df["rating"] > 3).astype(float)

    # Train / validation split (80/20)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = RatingDataset(
        df["user_idx"].values[train_idx],
        df["item_idx"].values[train_idx],
        df["label"].values[train_idx],
    )
    val_ds = RatingDataset(
        df["user_idx"].values[val_idx],
        df["item_idx"].values[val_idx],
        df["label"].values[val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = NCFModel(n_users, n_items, emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for u, i, y in train_loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(u, i)
            loss  = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(train_ds)

        # ── Validate ──
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for u, i, y in val_loader:
                u, i, y = u.to(device), i.to(device), y.to(device)
                preds = model(u, i)
                val_loss_total += criterion(preds, y).item() * len(y)
        val_loss = val_loss_total / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"✅  NCF training complete. Best val_loss={best_val_loss:.4f}")
    print(f"    Model saved to: {save_path}")

    # Load best weights
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


# ─── Inference wrapper ────────────────────────────────────────────────────────

class NCFRecommender:
    """Thin wrapper for fast inference in server.py (model is pre-trained)."""

    def __init__(self, model: NCFModel, n_users: int, n_items: int, device: str = "cpu"):
        self.model   = model.to(device)
        self.model.eval()
        self.n_users = n_users
        self.n_items = n_items
        self.device  = device

    def score_all_items(self, user_idx: int) -> np.ndarray:
        """Return predicted preference score for every item for this user."""
        if user_idx >= self.n_users:
            return np.zeros(self.n_items)

        u_tensor = torch.full((self.n_items,), user_idx, dtype=torch.long, device=self.device)
        i_tensor = torch.arange(self.n_items, dtype=torch.long, device=self.device)

        with torch.no_grad():
            scores = self.model(u_tensor, i_tensor).cpu().numpy()
        return scores

    def recommend(
        self, user_idx: int, n: int = 10,
        rated_items: set = None,
    ) -> list:
        scores  = self.score_all_items(user_idx)
        ranked  = np.argsort(scores)[::-1]
        results = []
        for i in ranked:
            if rated_items and i in rated_items:
                continue
            expl = (
                f"Recommended because similar users purchased this item; deep neural patterns "
                f"suggest high compatibility (NCF Score: {scores[i]:.3f})."
            )
            results.append((int(i), float(scores[i]), expl))
            if len(results) == n:
                break
        return results

    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str, n_users: int, n_items: int,
        emb_dim: int = 64, device: str = "cpu"
    ) -> "NCFRecommender":
        model = NCFModel(n_users, n_items, emb_dim)
        
        try:
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location=device)
            
        # Handle weights saved with DataParallel (removes 'module.' prefix)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        return cls(model, n_users, n_items, device)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train NCF model offline")
    parser.add_argument("--data",       default="data/processed", help="Processed data dir")
    parser.add_argument("--output",     default="models/saved/ncf_model.pt")
    parser.add_argument("--emb_dim",    type=int, default=64)
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr",         type=float, default=0.001)
    args = parser.parse_args()

    data_dir = Path(args.data)
    df_cf    = pd.read_parquet(data_dir / "cf_interactions.parquet")

    with open(data_dir / "mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model, history = train_ncf(
        df_cf,
        n_users    = mappings["n_users"],
        n_items    = mappings["n_items"],
        emb_dim    = args.emb_dim,
        lr         = args.lr,
        n_epochs   = args.epochs,
        batch_size = args.batch_size,
        save_path  = args.output,
    )
    print("Done.")
