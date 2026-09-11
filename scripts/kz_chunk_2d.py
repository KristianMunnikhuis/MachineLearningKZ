"""Generate ONE chunk of 2D KZ quench samples.

Run from the repo root:
    python -m scripts.kz_chunk_2d 7      # computes chunk 7

Each chunk uses its chunk number as the random seed, so every chunk is
independent and reproducible. If the output file already exists, it does nothing.
"""
import os
import sys
import numpy as np
import src.KZ_2D as kz2

# ---- physics parameters ----
kz2.dx    = 0.5
kz2.eta   = 1.0
kz2.theta = 1e-8
kz2.EPS_I = -1.5
kz2.EPS_F = 1.0
kz2.N     = 128        # grid sites per side
kz2.TAU   = 128.0      # quench time

# ---- run settings ----
N_STEPS     = 3200     # dt = 0.1 at TAU = 128
N_PER_CHUNK = 50       # samples in this chunk
CENTERS     = [20, 50, 70]
HALF_WIN    = 5        # save snapshots at center-5, ..., center+5
OUTDIR      = "data/2D"

# ---- which chunk is this? ----
chunk_id = int(sys.argv[1])
outfile  = os.path.join(OUTDIR, f"chunk_{chunk_id:03d}.npz")

if os.path.exists(outfile):
    print("already done:", outfile)
    sys.exit()

# ---- times at which to save snapshots: 15..25, 45..55, 65..75 ----
snap_times = []
for c in CENTERS:
    for k in range(-HALF_WIN, HALF_WIN + 1):
        snap_times.append(c + k)

# ---- time grid ----
t_hist, dt = kz2.make_time_grid(n_steps=N_STEPS)
t0, t1 = t_hist[0], t_hist[-1]

# step number corresponding to each snapshot time
snap_steps = [int(round((t - t0) / dt)) for t in snap_times]
assert all(1 <= j <= len(t_hist) - 1 for j in snap_steps), "snapshot time outside run"

# ---- initial state ----
rng   = np.random.default_rng()
shape = (N_PER_CHUNK, kz2.N, kz2.N)
phi   = np.zeros(shape)
pi    = np.zeros(shape)
noise_amp = np.sqrt(2 * kz2.eta * kz2.theta * dt / kz2.dx**2)

snaps = np.zeros((N_PER_CHUNK, len(snap_times), kz2.N, kz2.N), dtype=np.float32)

# ---- time evolution ----
for step in range(1, len(t_hist)):
    t = t_hist[step - 1]
    phi, pi = kz2.rk4_step(phi, pi, t, dt, t0, t1)
    pi += noise_amp * rng.standard_normal(shape)

    for i, j in enumerate(snap_steps):
        if j == step:
            snaps[:, i] = phi

# ---- save (write to temp file, then rename, so a crash never leaves a half file) ----
os.makedirs(OUTDIR, exist_ok=True)
tmpfile = os.path.join(OUTDIR, f"tmp_chunk_{chunk_id:03d}.npz")

np.savez(tmpfile,
         snaps=snaps,                          # (50, 33, N, N)
         snap_times=np.array(snap_times),      # (33,)
         phi_final=phi.astype(np.float32),     # (50, N, N)
         wall_len=kz2.wall_length(phi),        # (50,)
         seed=chunk_id, dt=dt, tau=kz2.TAU, N=kz2.N)

os.replace(tmpfile, outfile)
print("saved:", outfile)