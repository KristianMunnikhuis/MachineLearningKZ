"""Generate ONE chunk of 2D KZ quench samples.

Run from the repo root:
    python -m scripts.kz_chunk_2d 7             # chunk 7, default tau
    python -m scripts.kz_chunk_2d 7 --tau 256   # chunk 7, tau = 256

Each tau writes to its own folder, e.g. data/2D_tau128/chunk_007.npz.
Each chunk gets a fresh random seed, which is saved in the output file.
If the output file already exists, it does nothing.
"""
import os
import argparse
import numpy as np
import src.KZ_2D as kz2

# ---- command line ----
p = argparse.ArgumentParser()
p.add_argument("chunk_id", type=int)
p.add_argument("--tau",    type=float, default=128.0)
p.add_argument("--steps",  type=int,   default=None, help="default: 25*tau")
p.add_argument("--n",      type=int,   default=50,   help="samples per chunk")
p.add_argument("--outdir", type=str,   default=None, help="default: data/2D_tau<TAU>")
a = p.parse_args()

# ---- physics parameters ----
kz2.dx    = 0.5
kz2.eta   = 1.0
kz2.theta = 1e-8
kz2.EPS_I = -1.5
kz2.EPS_F = 1.0
kz2.N     = 128
kz2.TAU   = a.tau

# ---- run settings ----
N_STEPS     = a.steps if a.steps else int(round(25 * a.tau))   # keeps dt = 0.1
N_PER_CHUNK = a.n
FRACS      = np.linspace(-0.5, 1.0, 31)        # in units of tau, -0.5*tau .. tau
SNAP_TIMES = [float(f * a.tau) for f in FRACS]
OUTDIR      = a.outdir if a.outdir else f"data/2D_tau{int(a.tau)}"

outfile = os.path.join(OUTDIR, f"chunk_{a.chunk_id:03d}.npz")
if os.path.exists(outfile):
    print("already done:", outfile)
    raise SystemExit

# ---- time grid ----
t_hist, dt = kz2.make_time_grid(n_steps=N_STEPS)
t0, t1 = t_hist[0], t_hist[-1]
print(f"tau={a.tau}  steps={N_STEPS}  dt={dt:.4f}  t: {t0:.1f} .. {t1:.1f}", flush=True)

snap_steps = [int(round((t - t0) / dt)) for t in SNAP_TIMES]
assert all(1 <= j <= len(t_hist) - 1 for j in snap_steps), "snapshot time outside run"

# ---- initial state ----
seed = np.random.SeedSequence().entropy
rng  = np.random.default_rng(seed)
shape = (N_PER_CHUNK, kz2.N, kz2.N)
phi   = np.zeros(shape)
pi    = np.zeros(shape)
noise_amp = np.sqrt(2 * kz2.eta * kz2.theta * dt / kz2.dx**2)

snaps = np.zeros((N_PER_CHUNK, len(SNAP_TIMES), kz2.N, kz2.N), dtype=np.float32)
step_to_slot = {j: i for i, j in enumerate(snap_steps)}

# ---- time evolution ----
for step in range(1, len(t_hist)):
    t = t_hist[step - 1]
    phi, pi = kz2.rk4_step(phi, pi, t, dt, t0, t1)
    pi += noise_amp * rng.standard_normal(shape)

    if step in step_to_slot:
        snaps[:, step_to_slot[step]] = phi

# ---- save (write to temp file, then rename, so a crash never leaves a half file) ----
os.makedirs(OUTDIR, exist_ok=True)
tmpfile = os.path.join(OUTDIR, f"tmp_chunk_{a.chunk_id:03d}.npz")

np.savez(tmpfile,
         snaps=snaps,
         snap_times=np.array(SNAP_TIMES),
         phi_final=phi.astype(np.float32),
         wall_len=kz2.wall_length(phi),
         seed=str(seed), dt=dt, tau=a.tau, N=kz2.N, n_steps=N_STEPS)

os.replace(tmpfile, outfile)
print("saved:", outfile)