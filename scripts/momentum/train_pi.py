"""Momentum experiment (for fun): train the U-Net on the momentum pi ALONE.

Same samples, split, and training as train.py; only the input changes.
Run from the repo root:
    python -m momentum_exp.train_pi --t 3 --eta 0.3
Output: momentum_exp/results_eta<ETA>/t<t>_pi.json (and .pt)
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
p.add_argument("--t",   type=float, required=True, help="input time, units of t_hat")
p.add_argument("--eta", type=float, default=1.0)
a = p.parse_args()

DATA   = f"data/momentum/eta{a.eta:g}"
OUTDIR = f"results/momentum/eta{a.eta:g}"
# ---- load pi (input) and phi (only for the persistence baseline) ----
files = sorted(glob.glob(os.path.join(DATA, "chunk_*.npz")))
i = int(np.argmin(np.abs(np.load(files[0])["times"] - a.t)))
X   = np.concatenate([np.load(f)["pi"][:, i][:, None]  for f in files])     # (n, 1, N, N)
PHI = np.concatenate([np.load(f)["phi"][:, i]          for f in files])     # (n, N, N)
Y   = np.concatenate([np.load(f)["y_end"]              for f in files]).astype(np.float32)
print(f"eta = {a.eta:g}, t = {a.t:g} t_hat, input = pi only, X {X.shape}, device = {kzml.DEVICE}", flush=True)

# ---- same split as train.py; rescale pi by its training-set std ----
idx = np.random.default_rng(SEED).permutation(len(X))
n10 = len(X) // 10
test_idx, val_idx, train_idx = idx[:n10], idx[n10:2 * n10], idx[2 * n10:]
scale = X[train_idx].std()

def loader(ids, shuffle):
    ds = TensorDataset(torch.from_numpy(X[ids] / scale), torch.from_numpy(Y[ids][:, None]))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

baseline = float(((PHI[test_idx] > 0) == (Y[test_idx] > 0.5)).mean())     # persistence: sign of phi

# ---- train, then score the best checkpoint ----
os.makedirs(OUTDIR, exist_ok=True)
tag  = f"t{a.t:g}_pi"
ckpt = os.path.join(OUTDIR, tag + ".pt")
model = kzml.UNet(in_ch=1).to(kzml.DEVICE)
history, best_val = kzml.train(model, loader(train_idx, True), loader(val_idx, False),
                               epochs=EPOCHS, lr=LR, ckpt=ckpt)
model.load_state_dict(torch.load(ckpt, map_location=kzml.DEVICE))
_, test_acc = kzml.evaluate(model, loader(test_idx, False))

json.dump(dict(eta=a.eta, t=a.t, channels="pi", test_acc=test_acc, baseline=baseline, best_val=best_val,
               scale=float(scale), n_samples=len(X), epochs=EPOCHS, history=history),
          open(os.path.join(OUTDIR, tag + ".json"), "w"), indent=2)
print(f"test error {1 - test_acc:.5f}   persistence error {1 - baseline:.5f}", flush=True)