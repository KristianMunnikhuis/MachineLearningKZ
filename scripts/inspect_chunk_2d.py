"""Quick look at one 2D KZ data chunk.

Plots N_SHOW random samples at a few snapshot times, plus the final state.
Each column shares a symmetric color scale, so panels in the same column are comparable.

    python -m scripts.inspect_chunk 0 --tau 128
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---- inputs ----
p = argparse.ArgumentParser()
p.add_argument("chunk_id", type=int)
p.add_argument("--tau", type=int, default=128)
a = p.parse_args()

N_SHOW    = 4             # number of samples (rows)
SNAP_IDX  = [5, 16, 27]   # snapshot indices to show (columns)
SEED      = 0             # fixed so the same samples come up each time
FONTSIZE  = 16

# ---- load ----
d = np.load(f"data/2D_tau{a.tau}/chunk_{a.chunk_id:03d}.npz")
snaps      = d["snaps"]         # (n_samples, n_times, N, N)
snap_times = d["snap_times"]
phi_final  = d["phi_final"]     # (n_samples, N, N)

for key in d.files:
    print(key, d[key].shape)

rng   = np.random.default_rng(SEED)
picks = rng.choice(len(snaps), size=N_SHOW, replace=False)

# ---- plot ----
n_cols = len(SNAP_IDX) + 1
fig, axes = plt.subplots(N_SHOW, n_cols, figsize=(4 * n_cols, 4 * N_SHOW))

for col in range(n_cols):
    # field for this column: a snapshot time, or the final state
    if col < len(SNAP_IDX):
        fields = snaps[picks, SNAP_IDX[col]]
        title  = rf"$t/\tau = {snap_times[SNAP_IDX[col]] / a.tau:.2f}$"
    else:
        fields = phi_final[picks]
        title  = "final"

    # symmetric color scale shared down the column (red = +, blue = -)
    M = np.abs(fields).max()

    for row in range(N_SHOW):
        ax = axes[row, col]
        ax.imshow(fields[row], cmap="RdBu_r", vmin=-M, vmax=M)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(title, fontsize=FONTSIZE)
        if col == 0:
            ax.set_ylabel(f"sample {picks[row]}", fontsize=FONTSIZE)

fig.suptitle(rf"$\tau = {a.tau}$, chunk {a.chunk_id}", fontsize=FONTSIZE + 4)
fig.tight_layout()

# ---- save ----
outdir = f"figures/tau_{a.tau}"
os.makedirs(outdir, exist_ok=True)
outfile = f"{outdir}/chunk_{a.chunk_id:03d}.png"
plt.savefig(outfile, dpi=200)
print("saved:", outfile)
print("samples:", picks)
print("wall lengths:", d["wall_len"][picks])