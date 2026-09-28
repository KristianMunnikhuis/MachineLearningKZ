"""Shared pieces for the 2D KZ predictability experiments:
data loading, U-Net model, training and evaluation.

Import from a script or a notebook:
    import src.kz_ml as kzml
    X, Y, t_used = kzml.load_data("data/2D_tau128", t_in=0.0)
"""
import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time


DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

#Loss Function:
LOSS_FN = nn.BCEWithLogitsLoss()


# ---------------------------------------------------------------- data

def load_data(data_dir, t_in):
    """Build (input, target) pairs from 2D KZ quench chunks.

    Input  X: field snapshot at the stored time nearest t_in, shape (n, N, N)
    Target Y: final domain map, 1 where phi_final > 0 else 0, shape (n, N, N)
    Also returns t_used, the snapshot time actually used.
    Assumes all chunks share the same snap_times.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "chunk_*.npz")))
    if not files:
        raise FileNotFoundError(f"no chunks in {data_dir}")

    X_list, Y_list = [], []
    for f in files:
        d = np.load(f)
        # index of stored snapshot closest to requested time
        i = int(np.argmin(np.abs(d["snap_times"] - t_in)))
        t_used = float(d["snap_times"][i])
        X_list.append(d["snaps"][:, i])       # (runs, N, N) at t_used
        Y_list.append(d["phi_final"] > 0)     # final sign = domain label

    X = np.concatenate(X_list).astype(np.float32)
    Y = np.concatenate(Y_list).astype(np.float32)
    return X, Y, t_used


def make_loaders(X, Y, batch_size=32, seed=0):
    """Split 80/10/10 into train/val/test and wrap in DataLoaders.

    Input is scaled by the training-set variance. 
    Returns the three loaders, the scale factor, and the persistence baseline accuracy on the test set.
    """
    rng = np.random.default_rng(seed) #Randomly Split the Training Data
    idx = rng.permutation(len(X)) #Randomly arrange the data
    n10 = len(X) // 10 #Split into 10 groups
    test_idx, val_idx, train_idx = idx[:n10], idx[n10:2*n10], idx[2*n10:] #Split into 10/10/80 grouping

    scale = float(X[train_idx].std()) #Calculate STD of training data

    def loader(ids, shuffle):
        Xt = torch.from_numpy(X[ids][:, None] / scale) #Creates X for sample, adds channel axis, and scales by STD
        Yt = torch.from_numpy(Y[ids][:, None]) #Same with Y
        return DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size,
                          shuffle=shuffle) 
    #Check Baseline (How accurate you would be by assuming Post Quench evolution stops)
    baseline = float(((X[test_idx] > 0) == (Y[test_idx] > 0.5)).mean()) 
    
    return (loader(train_idx, True), loader(val_idx, False),
            loader(test_idx, False), scale, baseline)


# ---------------------------------------------------------------- model

def block(c_in, c_out):
    #Two 3x3 Convolutions (followed by ReLU) with periodic boundary conditions. 
    #Changes channel size
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular"), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode="circular"), nn.ReLU(),
    )


class UNet(nn.Module):
    #Basic UNet is enough to capture spatial data at all scales in the system
    def __init__(self, in_ch=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2)
        self.down1  = block(in_ch, 16)
        self.down2  = block(16, 32)
        self.down3  = block(32, 64)
        self.bottom = block(64, 128)
        self.up3 = block(128 + 64, 64)
        self.up2 = block(64 + 32, 32)
        self.up1 = block(32 + 16, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        #Downsample
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        b  = self.bottom(self.pool(d3))
        #Upsample with skip connections
        u3 = self.up3(torch.cat([self.up(b),  d3], dim=1))
        u2 = self.up2(torch.cat([self.up(u3), d2], dim=1))
        u1 = self.up1(torch.cat([self.up(u2), d1], dim=1))
        return self.out(u1)


# ---------------------------------------------------------------- train/eval

def evaluate(model, loader):
    """Mean loss and per-pixel accuracy over a loader."""
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        #Dont keep track of Gradients Here
        for xb, yb in loader: #Loops over Batches
            xb, yb = xb.to(DEVICE), yb.to(DEVICE) #sends batch to DEVICE
            logits = model(xb) #Forward pass data into model
            total_loss += LOSS_FN(logits, yb).item() * len(xb) 
            correct    += ((logits > 0) == (yb > 0.5)).sum().item()
            count      += yb.numel()
    return total_loss / len(loader.dataset), correct / count


def train(model, train_loader, val_loader, epochs=30, lr=1e-3,
          ckpt=None, verbose=True):
    """Train, saving the best-on-validation weights to ckpt.

    Returns (history, best_val_accuracy). history rows are
    (train_loss, val_loss, train_acc, val_acc).
    """

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history, best_val = [], 0.0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = LOSS_FN(model(xb), yb) #Calculate Loss
            opt.zero_grad() #Clear Gradients
            loss.backward() #BackProp
            opt.step() #Update Weights

        train_loss, train_acc = evaluate(model, train_loader) #Training loss
        val_loss,   val_acc   = evaluate(model, val_loader) #Validation Loss
        history.append((train_loss, val_loss, train_acc, val_acc)) #Append History

        saved = ""
        if val_acc > best_val:
            best_val = val_acc #Save best Validation Loss parameters
            if ckpt:
                torch.save(model.state_dict(), ckpt) #Save Model
                saved = "  *saved"

        if verbose: #Print information about the training
            print(f"epoch {epoch:2d}  train {train_acc:.4f}  val {val_acc:.4f}"
                  f"  ({time.time()-t0:.0f}s){saved}", flush=True)

    return history, best_val #Return History and Best Validation attempt