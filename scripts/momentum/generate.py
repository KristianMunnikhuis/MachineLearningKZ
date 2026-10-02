"""Momentum experiment: generate one chunk, saving phi and pi at multiples of SAVE_HAT * t_hat.

Run from the repo root:
    python -m momentum_exp.generate 0 --eta 0.3        # chunk 0, damping eta = 0.3
Output: momentum_exp/data_eta<ETA>/chunk_<ID>.npz
"""
import os
import argparse
import numpy as np
import src.kz_2d as kz2

# ---- inputs ----
TAU       = 4
TIMES     = [0, 0.5, 1, 2, 3, 4, 5, 6, 7]  # input times to keep, units of t_hat (multiples of SAVE_HAT)
SAVE_HAT  = 0.5                            # save spacing, units of t_hat
N         = 256
N_REAL    = 50                             # samples per chunk
DT        = 0.25                           # target step size (adjusted slightly to fit the save spacing)
T_END_HAT = 10.0                           # end of run = target time, units of t_hat

p = argparse.ArgumentParser()
p.add_argument("chunk_id", type=int)
p.add_argument("--eta", type=float, default=1.0, help="damping")
a = p.parse_args()

ETA    = a.eta
OUTDIR = f"data/momentum/eta{ETA:g}"
kz2.dx, kz2.eta, kz2.theta, kz2.N, kz2.TAU = 0.5, ETA, 1e-8, N, float(TAU)
t_hat = np.sqrt(2 * 1.0 * TAU)             # time unit fixed at its eta = 1 value, for comparison across eta

# time grid: start and end on multiples of SAVE_HAT * t_hat, so saved steps land exactly on them
save    = SAVE_HAT * t_hat
stride  = int(round(save / DT))
dt      = save / stride
t_start = -np.ceil(TAU / save) * save
t_end   = T_END_HAT * t_hat
n_steps = int(round((t_end - t_start) / dt))
kz2.EPS_I, kz2.EPS_F = t_start / TAU, t_end / TAU     # eps = t / tau throughout

t_hist, dt = kz2.make_time_grid(n_steps=n_steps)
phi_hist, pi_hist = kz2.run(t_hist, dt, n_real=N_REAL, stride=stride)   # (n_saved, N_REAL, N, N)

t_saved = t_hist[::stride] / t_hat
keep    = [int(np.argmin(np.abs(t_saved - t))) for t in TIMES]

os.makedirs(OUTDIR, exist_ok=True)
np.savez(os.path.join(OUTDIR, f"chunk_{a.chunk_id:03d}.npz"),
         phi=phi_hist[keep].transpose(1, 0, 2, 3).astype(np.float32),     # (N_REAL, len(TIMES), N, N)
         pi=pi_hist[keep].transpose(1, 0, 2, 3).astype(np.float32),
         y_end=phi_hist[-1] > 0,
         times=t_saved[keep], tau=TAU, t_hat=t_hat, eta=ETA)
print("saved chunk", a.chunk_id, " eta =", ETA, " times kept:", np.round(t_saved[keep], 3))