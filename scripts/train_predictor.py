from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("Install with:  pip install torch")
        sys.exit(1)

    from vision.predictor_model import LSTMRegressor

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/trajectories.npz")
    parser.add_argument("--out", default="data/lstm_predictor.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-noise", type=float, default=0.005,
                        help="Gaussian noise on the input window (m) to "
                             "match EKF posterior statistics at deploy time.")
    args = parser.parse_args()

    blob = np.load(ROOT / args.data)
    X = blob["X"]; Y = blob["Y"]
    window = int(blob["window"]); horizon = int(blob["horizon"])
    print(f"Loaded {X.shape[0]} pairs.  window={window}, horizon={horizon}")

    mean_z = X.reshape(-1, 6).mean(0)
    std_z  = X.reshape(-1, 6).std(0) + 1e-6
    X_norm = (X - mean_z) / std_z

    rng = np.random.default_rng(0)
    idx = rng.permutation(X.shape[0])
    split = int(0.9 * X.shape[0])
    tr, va = idx[:split], idx[split:]

    Xt = torch.from_numpy(X_norm[tr]).float()
    Yt = torch.from_numpy(Y[tr]).float()
    Xv = torch.from_numpy(X_norm[va]).float()
    Yv = torch.from_numpy(Y[va]).float()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {dev}")
    model = LSTMRegressor(input_dim=6, hidden=args.hidden,
                          n_horizons=horizon).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = TensorDataset(Xt, Yt)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    def gaussian_nll(mu, log_sigma, target):
        var = torch.exp(2.0 * log_sigma)
        nll = 0.5 * ((target - mu) ** 2 / var + 2.0 * log_sigma
                     + np.log(2.0 * np.pi))
        return nll.mean()

    best_val = float("inf")
    history = {"epoch": [], "val_nll": [], "val_mae_mm": []}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve_path = ROOT / "data" / "training_curve.png"
    curve_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(dev); yb = yb.to(dev)
            xb_noisy = xb + torch.randn_like(xb) * (args.input_noise / std_z[0])
            mu, log_sigma = model(xb_noisy)
            loss = gaussian_nll(mu, log_sigma, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            mu_v, log_sigma_v = model(Xv.to(dev))
            val_loss = gaussian_nll(mu_v, log_sigma_v, Yv.to(dev)).item()
            mae_mm = (mu_v.cpu() - Yv).abs().mean().item() * 1000
        print(f"epoch {epoch+1:3d}/{args.epochs}   "
              f"val_nll={val_loss:7.3f}   val_mae={mae_mm:6.1f} mm", flush=True)

        history["epoch"].append(epoch + 1)
        history["val_nll"].append(val_loss)
        history["val_mae_mm"].append(mae_mm)

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(history["epoch"], history["val_nll"], "C0o-", label="val NLL")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("val NLL", color="C0")
        ax1.grid(alpha=.3)
        ax2 = ax1.twinx()
        ax2.plot(history["epoch"], history["val_mae_mm"], "C3s-", label="val MAE (mm)")
        ax2.set_ylabel("val MAE (mm)", color="C3")
        ax1.set_title(f"LSTM predictor training — epoch {epoch+1}/{args.epochs}")
        fig.tight_layout()
        fig.savefig(curve_path, dpi=110)
        plt.close(fig)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "input_dim": 6, "hidden": args.hidden,
                "n_horizons": horizon, "window": window,
                "mean_z": mean_z, "std_z": std_z,
            }, ROOT / args.out)
    print(f"\nBest val NLL: {best_val:.3f}.   Checkpoint -> {ROOT / args.out}")
    print(f"Training curve: {curve_path}")


if __name__ == "__main__":
    main()
