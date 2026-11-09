"""
Model Training & Checkpoint Engine
Electricity Demand Forecasting Using Deep Learning: A Comparative Study of 1D CNN, LSTM and CNN–LSTM Models
"""

import os
import sys
import time
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def train_single_model(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu",
    checkpoint_path: Optional[str] = None
) -> Tuple[Any, Dict[str, list]]:
    """
    Standard PyTorch training loop with AdamW, ReduceLROnPlateau, and Early Stopping.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Run on Google Colab or install torch.")

    model = model.to(device)
    criterion = nn.HuberLoss() # Robust to extreme load spikes
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"Beginning training on device: {device} | Max Epochs: {epochs} | Early Stopping Patience: {patience}")
    sys.stdout.flush()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss_accum = 0.0
        n_train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item()
            n_train_batches += 1

        avg_train_loss = train_loss_accum / max(n_train_batches, 1)

        # 2. Validation Phase
        model.eval()
        val_loss_accum = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss_accum += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss_accum / max(n_val_batches, 1)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")
            sys.stdout.flush()

        # Early Stopping & Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_weights, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (Best Val Loss: {best_val_loss:.6f})")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history
