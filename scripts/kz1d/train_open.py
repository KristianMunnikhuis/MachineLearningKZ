"""Train the Suzuki RNN for one (case, tau, input time) and save the metrics.

Model, window and target are identical to notebooks 01/05:
11-snapshot window (spacing 1) centered at t -> SimpleRNN(256, softsign) -> Dense(DOF) -> phi(t = tau), MSE.
Input normalized by the training-set std. Keras validation_split = last 20% of samples.
Early stopping on val_loss, best weights restored.

Example usage (from the repo root):
    python -m scripts.kz1d.train_open twa 128 -100

Output: results/open/{case}_tau{TAU}_t{t}.json
"""
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")    # set KERAS_BACKEND=torch if tensorflow is missing
import argparse, glob, json, time
import numpy as np
import keras
import src.kz as kz

# ---- parameters ----
WIN      = 5            # half-width: 2*WIN+1 snapshots
EPOCHS   = 150
PATIENCE = 15
BATCH    = 10
SEED     = 0
DATADIR  = "data/1D_open"
OUTDIR   = "results/open"


def main(a):
    t0_wall = time.time()
    keras.utils.set_random_seed(a.seed)

    files = sorted(glob.glob(os.path.join(a.datadir, a.case, f"tau{int(a.tau)}", "chunk_*.npz")))
    if not files:
        raise FileNotFoundError(f"no chunks for {a.case} tau={a.tau} in {a.datadir}")

    # window indices from the shared snapshot grid
    ts  = np.load(files[0])["ts"]
    j   = int(np.argmin(np.abs(ts - a.t)))
    k   = int(round(1.0/(ts[1] - ts[0])))
    idx = np.arange(j - a.win*k, j + a.win*k + 1, k)
    if idx[0] < 0 or idx[-1] >= len(ts):
        raise ValueError(f"t={a.t} window runs off the stored range [{ts[0]:.0f}, {ts[-1]:.0f}]")

    X_list, Y_list = [], []
    for f in files:
        d = np.load(f)
        X_list.append(d["phi_hist"][:, idx, :])
        Y_list.append(d["phi_final"])
    X = np.concatenate(X_list).astype("float32")       # (N, 2*win+1, DOF)
    Y = np.concatenate(Y_list).astype("float32")       # (N, DOF)
    N, DOF  = Y.shape
    n_train = int(0.8*N)
    N_VAL   = N - n_train
    X = X/X[:n_train].std()
    print(f"{a.case} tau={a.tau} t={a.t}: {len(files)} chunks, X {X.shape}, Y {Y.shape}", flush=True)

    model = keras.Sequential([
        keras.layers.Input(shape=(2*a.win + 1, DOF)),
        keras.layers.SimpleRNN(256, activation="softsign"),
        keras.layers.Dense(DOF),
    ])
    model.compile(optimizer="adam", loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True)
    h = model.fit(X, Y, epochs=a.epochs, batch_size=a.batch, validation_split=0.2, verbose=2, callbacks=[stop])

    P      = model.predict(X[-N_VAL:], verbose=0)
    Y_val  = Y[-N_VAL:]
    n_true = kz.count_defects(Y_val.astype(np.float64))
    n_pred = kz.count_defects(P.astype(np.float64))
    res = dict(case=a.case, tau=a.tau, t=a.t, t_used=float(ts[j]), win=a.win, N=N, n_val=N_VAL,
               val_loss=float(min(h.history["val_loss"])),
               sign_err=float(((P > 0) != (Y_val > 0)).mean()),
               base_err=float(((X[-N_VAL:, a.win] > 0) != (Y_val > 0)).mean()),
               count_acc=float((n_pred == n_true).mean()),
               n_epochs=len(h.history["val_loss"]),
               loss_hist=[float(v) for v in h.history["loss"]],
               val_loss_hist=[float(v) for v in h.history["val_loss"]],
               seconds=time.time() - t0_wall)

    os.makedirs(a.outdir, exist_ok=True)
    path = os.path.join(a.outdir, f"{a.case}_tau{int(a.tau)}_t{a.t:g}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"sign error = {res['sign_err']:.4f}   baseline = {res['base_err']:.4f}   "
          f"val MSE = {res['val_loss']:.4f}   epochs = {res['n_epochs']}   -> {path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case")
    p.add_argument("tau", type=float)
    p.add_argument("t", type=float)
    p.add_argument("--win", type=int, default=WIN)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--datadir", default=DATADIR)
    p.add_argument("--outdir", default=OUTDIR)
    main(p.parse_args())
