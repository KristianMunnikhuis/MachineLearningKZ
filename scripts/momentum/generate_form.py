"""Momentum experiment, formation target: generate one chunk saving phi and pi at chosen times,
plus the sign map at formation (y_form) and at the end (y_end).

Time unit t_hat = sqrt(2*tau) is fixed at its eta = 1 value, as in generate.py.
Formation, per sample: first time phi_rms >= FRAC*sqrt(eps).

Run from the repo root:
    python -m scripts.momentum.generate_form 0 --eta 0.3
Output: data/momentum_form/eta<ETA>/chunk_<ID>.npz  (skips if it already exists)
"""
import os
import argparse
import numpy as np
import src.kz_2d as kz2

p = argparse.ArgumentParser()
p.add_argument("chunk_id", type=int)
p.add_argument("--eta", type=float, default=1.0, help="damping")
p.add_argument("--n",   type=int,   default=50,  help="samples per chunk")
a = p.parse_args()

# ---- inputs ----
TAU       = 4
TIMES     = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5])   # input times to save, units of t_hat
N         = 256
DT        = 0.25
FRAC      = 0.5                                     # formation threshold: phi_rms >= FRAC*sqrt(eps)
T_END_HAT = 10.0                                    # end of run, units of t_hat
ETA       = a.eta
OUTDIR    = f"data/momentum_form/eta{ETA:g}"

kz2.dx, kz2.eta, kz2.theta, kz2.N, kz2.TAU = 0.5, ETA, 1e-8, N, float(TAU)
t_hat = np.sqrt(2 * 1.0 * TAU)
t_max = T_END_HAT * t_hat
kz2.EPS_I = -1.0
kz2.EPS_F = t_max / TAU

outfile = os.path.join(OUTDIR, f"chunk_{a.chunk_id:03d}.npz")
if os.path.exists(outfile):
    print("already done:", outfile)
    raise SystemExit

# ---- time grid ----
n_steps = int(round((kz2.EPS_F - kz2.EPS_I) * TAU / DT))
t_hist, dt = kz2.make_time_grid(n_steps=n_steps)
t0, t1 = t_hist[0], t_hist[-1]
step_to_slot = {int(round((s * t_hat - t0) / dt)): i for i, s in enumerate(TIMES)}

# ---- initial state ----
seed  = np.random.SeedSequence().entropy
rng   = np.random.default_rng(seed)
shape = (a.n, N, N)
phi, pi = np.zeros(shape), np.zeros(shape)
amp   = np.sqrt(2 * ETA * kz2.theta * dt / kz2.dx**2)

phi_s  = np.zeros((a.n, len(TIMES), N, N), dtype=np.float32)
pi_s   = np.zeros((a.n, len(TIMES), N, N), dtype=np.float32)
y_form = np.zeros(shape, dtype=bool)
t_form = np.full(a.n, np.nan)

# ---- time evolution ----
for step in range(1, len(t_hist)):
    phi, pi = kz2.rk4_step(phi, pi, t_hist[step - 1], dt, t0, t1)
    pi += amp * rng.standard_normal(shape)
    t   = t_hist[step]
    eps = t / TAU

    if step in step_to_slot:
        phi_s[:, step_to_slot[step]] = phi
        pi_s[:, step_to_slot[step]]  = pi

    if eps > 0:
        rms = np.sqrt((phi**2).mean(axis=(1, 2)))
        new = np.isnan(t_form) & (rms >= FRAC * np.sqrt(eps))
        y_form[new] = phi[new] > 0
        t_form[new] = t

assert np.isfinite(phi).all(), "non-finite values; reduce dt"
if np.isnan(t_form).any():
    print(f"WARNING: {np.isnan(t_form).sum()} samples never reached formation", flush=True)

# ---- save ----
os.makedirs(OUTDIR, exist_ok=True)
tmpfile = os.path.join(OUTDIR, f"tmp_chunk_{a.chunk_id:03d}.npz")
np.savez(tmpfile, phi=phi_s, pi=pi_s, times=TIMES, y_form=y_form, t_form=t_form, y_end=(phi > 0),
         tau=TAU, t_hat=t_hat, eta=ETA, N=N, dx=kz2.dx, seed=str(seed))
os.replace(tmpfile, outfile)
print(f"saved {outfile}   t_form/t_hat = {np.nanmean(t_form) / t_hat:.2f} ± {np.nanstd(t_form) / t_hat:.2f}")