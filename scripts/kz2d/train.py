"""Train one U-Net: one tau, one input time, one target (v2 data).

Input time is in units of t_hat = sqrt(2*eta*tau); the nearest stored snapshot is used.
Target is the sign map at the end of the run (y_end, coarsened) or at formation (y_form, KZ).

Run from the repo root:
    python -m scripts.kz2d.train --tau 128 --t-hat 4
    python -m scripts.kz2d.train --tau 128 --t-hat 4 --target y_form --epochs 50

Writes results/v2/tau128_that4_y_end.json and .pt
"""
import argparse
import json
import os
import torch
import src.kz_ml as kzml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau",    type=float, required=True)
    p.add_argument("--t-hat",  type=float, required=True,
                   help="input time in units of t_hat; nearest stored snapshot is used")
    p.add_argument("--target", type=str,   default="y_end", choices=["y_end", "y_form"])
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int,   default=32)
    p.add_argument("--seed",   type=int,   default=0, help="split seed")
    p.add_argument("--data",   type=str,   default=None, help="default: data/2D_v2/tau<TAU>")
    p.add_argument("--outdir", type=str,   default="results/v2")
    return p.parse_args()


def main():
    a = parse_args()
    data_dir = a.data if a.data else f"data/2D_v2/tau{int(a.tau)}"
    os.makedirs(a.outdir, exist_ok=True)

    tag  = f"tau{int(a.tau)}_that{a.t_hat:g}_{a.target}"
    ckpt = os.path.join(a.outdir, tag + ".pt")
    print(f"device: {kzml.DEVICE}", flush=True)

    # load data: input snapshot at t_hat, chosen target
    X, Y, t_used = kzml.load_data(data_dir, a.t_hat, target=a.target)
    print(f"{data_dir}: {X.shape[0]} samples, t/t_hat = {t_used:g}, target = {a.target}", flush=True)

    # split 80/10/10, normalize, persistence baseline on the test set
    train_loader, val_loader, test_loader, scale, baseline = kzml.make_loaders(
        X, Y, batch_size=a.batch, seed=a.seed)
    print(f"baseline test accuracy: {baseline:.4f}", flush=True)

    # train from scratch; best-on-validation weights saved to ckpt
    model = kzml.UNet().to(kzml.DEVICE)
    history, best_val = kzml.train(model, train_loader, val_loader,
                                   epochs=a.epochs, lr=a.lr, ckpt=ckpt)

    # final number from the best checkpoint, on the untouched test set
    model.load_state_dict(torch.load(ckpt, map_location=kzml.DEVICE))
    test_loss, test_acc = kzml.evaluate(model, test_loader)

    # fraction of the baseline's errors the model removes (nan if baseline is perfect)
    error_cut = 1 - (1 - test_acc) / (1 - baseline) if baseline < 1 else float("nan")

    result = dict(tau=a.tau, t_hat_in=a.t_hat, t_used=t_used, target=a.target,
                  baseline=baseline, test_acc=test_acc, best_val=best_val, error_cut=error_cut,
                  epochs=a.epochs, lr=a.lr, batch=a.batch, seed=a.seed,
                  n_samples=int(X.shape[0]), scale=scale, history=history)

    with open(os.path.join(a.outdir, tag + ".json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"test {test_acc:.4f}  baseline {baseline:.4f}  "
          f"error cut {100 * error_cut:.1f}%", flush=True)


if __name__ == "__main__":
    main()