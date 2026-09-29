"""Generate ONE chunk of 2D KZ quench samples (v2: KZ-scaled times).

Times are measured in units of the freeze-out time t_hat = sqrt(2*eta*tau).
Each run goes from t = -tau (eps = -1) to t = T_MAX_HAT * t_hat, with eps = t/tau throughout.
Saved per sample:
    snaps    field at t/t_hat = SNAP_HAT (inputs)
    y_form   sign(phi) > 0 at formation, t_form = first time phi_rms >= FRAC*sqrt(eps)  (KZ target)
    y_end    sign(phi) > 0 at t = T_MAX_HAT * t_hat                                   (coarsened target)

Run from the repo root:
    python -m scripts.kz2d.generate_data 0 --tau 32          # chunk 0
    python -m scripts.kz2d.generate_data 0 --tau 32 --n 4    # quick pilot
Output: data/2D_v2/tau<TAU>/chunk_<ID>.npz  (skips if it already exists)
"""
import os
import argparse
import numpy as np
import src.kz_2d as kz2

# ---- command line ----
p = argparse.ArgumentParser()
p.add_argument("chunk_id", type=int)
p.add_argument("--tau", type=float, required=True)
p.add_argument("--n",   type=int,   default=50, help="samples per chunk")
a = p.parse_args()

# ---- inputs ----
N         = 256                           # sites per side (box L = N*dx = 128)
DT        = 0.25                          # RK4 step (stability limit dx = 0.5)
FRAC      = 0.5                           # formation threshold: phi_rms >= FRAC*sqrt(eps)
T_MAX_HAT = 10.0                          # run until t = T_MAX_HAT * t_hat
SNAP_HAT  = np.arange(0.0, T_MAX_HAT + 1e-9, 0.5)   # snapshot times in units of t_hat (21 snapshots)
OUTDIR    = f"data/2D_v2/tau{int(a.tau)}"

kz2.dx, kz2.eta, kz2.theta = 0.5, 1.0, 1e-8
kz2.N   = N
kz2.TAU = a.tau

t_hat  = np.sqrt(2 * kz2.eta * a.tau)
xi_hat = np.sqrt(2) * (a.tau / (2 * kz2.eta))**0.25
t_max  = T_MAX_HAT * t_hat

# eps = t/tau from t = -tau to t = t_max  (EPS_F set so the run ends at t_max)
kz2.EPS_I = -1.0
kz2.EPS_F = t_max / a.tau

outfile = os.path.join(OUTDIR, f"chunk_{a.chunk_id:03d}.npz")
if os.path.exists(outfile):
    print("already done:", outfile)
    raise SystemExit

# ---- time grid ----
n_steps = int(round((kz2.EPS_F - kz2.EPS_I) * a.tau / DT))
t_hist, dt = kz2.make_time_grid(n_steps=n_steps)
t0, t1 = t_hist[0], t_hist[-1]
snap_steps = [int(round((s * t_hat - t0) / dt)) for s in SNAP_HAT]
step_to_slot = {j: i for i, j in enumerate(snap_steps)}
print(f"tau={a.tau}  t_hat={t_hat:.2f}  xi_hat={xi_hat:.2f}  eps_F={kz2.EPS_F:.2f}  "
      f"steps={n_steps}  dt={dt:.3f}", flush=True)

# ---- initial state ----
seed = np.random.SeedSequence().entropy
rng  = np.random.default_rng(seed)
shape = (a.n, N, N)
phi = np.zeros(shape)
pi  = np.zeros(shape)
amp = np.sqrt(2 * kz2.eta * kz2.theta * dt / kz2.dx**2)

snaps  = np.zeros((a.n, len(SNAP_HAT), N, N), dtype=np.float16)
y_form = np.zeros(shape, dtype=bool)
t_form = np.full(a.n, np.nan)

# ---- time evolution ----
# custom loop instead of kz2.run(): saves only chosen snapshots + per-sample formation state
for step in range(1, len(t_hist)):
    phi, pi = kz2.rk4_step(phi, pi, t_hist[step - 1], dt, t0, t1)
    pi += amp * rng.standard_normal(shape)
    t   = t_hist[step]
    eps = t / a.tau

    if step in step_to_slot:
        snaps[:, step_to_slot[step]] = phi

    # per-sample formation: first time this sample's rms reaches FRAC*sqrt(eps)
    if eps > 0:
        rms = np.sqrt((phi**2).mean(axis=(1, 2)))
        new = np.isnan(t_form) & (rms >= FRAC * np.sqrt(eps))
        y_form[new] = phi[new] > 0
        t_form[new] = t

assert np.isfinite(phi).all(), "non-finite values; reduce dt"
if np.isnan(t_form).any():
    print(f"WARNING: {np.isnan(t_form).sum()} samples never reached formation", flush=True)

# ---- save (temp file then rename, so a crash never leaves a half-written chunk) ----
os.makedirs(OUTDIR, exist_ok=True)
tmpfile = os.path.join(OUTDIR, f"tmp_chunk_{a.chunk_id:03d}.npz")
np.savez(tmpfile,
         snaps=snaps, snap_hat=SNAP_HAT, snap_times=SNAP_HAT * t_hat,
         y_form=y_form, t_form=t_form, y_end=(phi > 0),
         t_hat=t_hat, xi_hat=xi_hat, tau=a.tau, N=N, dx=kz2.dx,
         dt=dt, frac=FRAC, t_max_hat=T_MAX_HAT, seed=str(seed))
os.replace(tmpfile, outfile)
print(f"saved: {outfile}   t_form/t_hat = {np.nanmean(t_form) / t_hat:.2f} ± {np.nanstd(t_form) / t_hat:.2f}")