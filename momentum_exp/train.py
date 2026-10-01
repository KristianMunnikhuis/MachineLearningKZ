"""Momentum experiment: train one U-Net on phi alone or on (phi, pi), predicting sign(phi) at the end.

Both variants use the same samples and the same split, so any difference comes from the momentum input.

Run from the repo root:
    python -m momentum_exp.train --t 3 --channels phi_pi --eta 0.3
Output: momentum_exp/results_eta<ETA>/t<t>_<channels>.json (and .pt)
"""
import os
import glob
import json
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import src.kz_ml as kzml

# ---- inputs ----
EPOCHS, LR, BATCH, SEED = 30, 1e-3, 32, 0

p = argparse.ArgumentParser()
p.add_argument("--t",        type=float, required=True, help="input time, units of t_hat")
p.add_argument("--channels", type=str,   required=True, choices=["phi", "phi_pi"])
p.add_argument("--eta",      type=float, default=1.0, help="damping (selects the data folder)")
a = p.parse_args()

DATA   = f"momentum_exp/data_eta{a.eta:g}"
OUTDIR = f"momentum_exp/results_eta{a.eta:g}"

# ---- load the chosen time from every chunk ----
files = sorted(glob.glob(os.path.join(DATA, "chunk_*.npz")))
i = int(np.argmin(np.abs(np.load(files[0])["times"] - a.t)))

X_list, Y_list = [], []
for f in files:
    d = np.load(f)
    x = d["phi"][:, i][:, None]                                    # (n, 1, N, N)
    if a.channels == "phi_pi":
        x = np.concatenate([x, d["pi"][:, i][:, None]], axis=1)    # (n, 2, N, N)
    X_list.append(x)
    Y_list.append(d["y_end"])
X = np.concatenate(X_list)
Y = np.concatenate(Y_list).astype(np.float32)
print(f"eta = {a.eta:g}, t = {a.t:g} t_hat, channels = {a.channels}, X {X.shape}, device = {kzml.DEVICE}", flush=True)

# ---- split 80/10/10 (same seed for both variants), scale each channel separately ----
idx = np.random.default_rng(SEED).permutation(len(X))
n10 = len(X) // 10
test_idx, val_idx, train_idx = idx[:n10], idx[n10:2 * n10], idx[2 * n10:]
scale = X[train_idx].std(axis=(0, 2, 3), keepdims=True)[0]         # (C, 1, 1)

def loader(ids, shuffle):
    ds = TensorDataset(torch.from_numpy(X[ids] / scale), torch.from_numpy(Y[ids][:, None]))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

baseline = float(((X[test_idx, 0] > 0) == (Y[test_idx] > 0.5)).mean())   # persistence: sign of phi

# ---- train ----
os.makedirs(OUTDIR, exist_ok=True)
tag  = f"t{a.t:g}_{a.channels}"
ckpt = os.path.join(OUTDIR, tag + ".pt")
model = kzml.UNet(in_ch=X.shape[1]).to(kzml.DEVICE)
history, best_val = kzml.train(model, loader(train_idx, True), loader(val_idx, False),
                               epochs=EPOCHS, lr=LR, ckpt=ckpt)

# ---- score the best checkpoint on the test set ----
model.load_state_dict(torch.load(ckpt, map_location=kzml.DEVICE))
_, test_acc = kzml.evaluate(model, loader(test_idx, False))

json.dump(dict(eta=a.eta, t=a.t, channels=a.channels, test_acc=test_acc, baseline=baseline, best_val=best_val,
               scale=scale.ravel().tolist(), n_samples=len(X), epochs=EPOCHS, history=history),
          open(os.path.join(OUTDIR, tag + ".json"), "w"), indent=2)
print(f"test error {1 - test_acc:.5f}   persistence error {1 - baseline:.5f}", flush=True)