"""Train one U-Net: one tau, one input time.

Run from the repo root:
    python -m scripts.train --tau 128 --t-in 50
    python -m scripts.train --tau 128 --t-in 50 --epochs 50

Writes results/tau128_t50.json and results/tau128_t50.pt
"""
import argparse
import json
import os
import torch
import src.kz_ml as kzml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tau",     type=float, required=True)
    p.add_argument("--t-in",    type=float, required=True,
                   help="input time; nearest available snapshot is used")
    p.add_argument("--epochs",  type=int,   default=30)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--batch",   type=int,   default=32)
    p.add_argument("--seed",    type=int,   default=0, help="split seed")
    p.add_argument("--data",    type=str,   default=None,
                   help="default: data/2D_tau<TAU>")
    p.add_argument("--outdir",  type=str,   default="results")
    return p.parse_args()


def main():

    a = parse_args() #training args
    data_dir = a.data if a.data else f"data/2D_tau{int(a.tau)}" #load data
    os.makedirs(a.outdir, exist_ok=True) #make output folders

    tag  = f"tau{int(a.tau)}_t{a.t_in:g}" #Tag Data
    ckpt = os.path.join(a.outdir, tag + ".pt") #Grab Checkpoints

    print(f"device: {kzml.DEVICE}", flush=True) #Print Device

    X, Y, t_used = kzml.load_data(data_dir, a.t_in) #Load Data
    print(f"{data_dir}: {X.shape[0]} samples, using t = {t_used:g}", flush=True)

    train_loader, val_loader, test_loader, scale, baseline = kzml.make_loaders(
        X, Y, batch_size=a.batch, seed=a.seed) # Check Baselines
    print(f"baseline test accuracy: {baseline:.4f}", flush=True)

    model = kzml.UNet().to(kzml.DEVICE) #Load NN 
    history, best_val = kzml.train(model, train_loader, val_loader,
                                   epochs=a.epochs, lr=a.lr, ckpt=ckpt) #Train (From checkpoint if available)
    # final number from the best checkpoint, on the untouched test set
    model.load_state_dict(torch.load(ckpt))
    test_loss, test_acc = kzml.evaluate(model, test_loader)
    # fraction of the baseline's errors the model removes (nan if baseline is perfect)
    error_cut = 1 - (1 - test_acc) / (1 - baseline) if baseline < 1 else float("nan")
    result = dict(tau=a.tau, t_in=a.t_in, t_used=t_used,
                  baseline=baseline, test_acc=test_acc, best_val=best_val,
                  error_cut=error_cut,
                  epochs=a.epochs, lr=a.lr, batch=a.batch, seed=a.seed,
                  n_samples=int(X.shape[0]), scale=scale, history=history)

    with open(os.path.join(a.outdir, tag + ".json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"test {test_acc:.4f}  baseline {baseline:.4f}  "
          f"error cut {100*result['error_cut']:.1f}%", flush=True)


if __name__ == "__main__":
    main()