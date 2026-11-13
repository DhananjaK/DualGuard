"""Step 3: Stage 1 – Autoencoder trained on BENIGN flows only.

Idea: learn to compress and rebuild normal traffic. A flow it rebuilds badly
(high reconstruction error) does not look normal -> anomaly.

    68 features ─► 64 ─► 32 ─► 16 (bottleneck) ─► 32 ─► 64 ─► 68 features
                  └──── encoder ────┘               └──── decoder ────┘
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], bottleneck: int):
        super().__init__()
        sizes = [n_features, *hidden, bottleneck]
        enc, dec = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        rev = sizes[::-1]
        for i, (a, b) in enumerate(zip(rev[:-1], rev[1:])):
            dec.append(nn.Linear(a, b))
            if i < len(rev) - 2:          # no activation on the output layer:
                dec.append(nn.ReLU())     # inputs are standardised, so can be negative
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        self.config = {"n_features": n_features, "hidden": list(hidden), "bottleneck": bottleneck}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


@torch.no_grad()
def reconstruction_error(model: Autoencoder, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    """Anomaly score per flow = mean squared difference between input and its rebuild."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch])
        out.append(((model(xb) - xb) ** 2).mean(dim=1).numpy())
    return np.concatenate(out)


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, p: dict, seed: int,
                      log=print) -> tuple[Autoencoder, list[dict]]:
    """Train on benign train rows; early-stop on benign validation loss."""
    torch.manual_seed(seed)
    torch.set_num_threads(p.get("threads", 8))
    model = Autoencoder(X_train.shape[1], p["hidden"], p["bottleneck"])
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"])
    gen = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X_train)

    best_loss, best_state, bad_epochs, history = np.inf, None, 0, []
    for epoch in range(1, p["epochs"] + 1):
        model.train()
        order = torch.randperm(len(Xt), generator=gen)     # reshuffle every epoch
        total = 0.0
        for i in range(0, len(Xt), p["batch_size"]):
            xb = Xt[order[i:i + p["batch_size"]]]
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(Xt)
        val_loss = float(reconstruction_error(model, X_val).mean())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log(f"epoch {epoch:2d}  train {train_loss:.5f}  val {val_loss:.5f}")

        if val_loss < best_loss - 1e-6:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= p["patience"]:
                log(f"early stop: no improvement for {p['patience']} epochs")
                break

    model.load_state_dict(best_state)     # keep the best epoch, not the last
    return model, history


def save_autoencoder(model: Autoencoder, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_autoencoder(path: Path) -> Autoencoder:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = Autoencoder(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
