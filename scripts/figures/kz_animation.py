import sys; sys.path.insert(0, "..")
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import src.kz_2d as kz2
# ---- inputs ----
TAUS     = [5, 20, 40]    # quench times (one panel each)
N        = 512            # sites per side
NFRAMES  = 120            # frames in the GIF
SEED     = 88
FPS      = 20
OUTFILE  = "../figures/2d/kz_quench.gif"

kz2.dx, kz2.eta, kz2.theta = 0.5, 1.0, 1e-8
kz2.EPS_I, kz2.EPS_F = -1.5, 1.0
kz2.N = N

runs = {}
for TAU in TAUS:
    kz2.TAU = float(TAU)
    t_hist, dt = kz2.make_time_grid(n_steps=int(round(25 * TAU)))   # dt = 0.1
    t0, t1 = t_hist[0], t_hist[-1]
    stride = max(1, (len(t_hist) - 1) // NFRAMES)

    rng = np.random.default_rng(SEED)
    phi = np.zeros((N, N))
    pi  = np.zeros((N, N))
    amp = np.sqrt(2 * kz2.eta * kz2.theta * dt / kz2.dx**2)

    frames, times = [], []
    for step in range(1, len(t_hist)):
        phi, pi = kz2.rk4_step(phi, pi, t_hist[step - 1], dt, t0, t1)
        pi += amp * rng.standard_normal(phi.shape)
        if step % stride == 0:
            frames.append(phi.astype(np.float32).copy())
            times.append(t_hist[step])

    runs[TAU] = (np.array(frames[:NFRAMES]), np.array(times[:NFRAMES]))
    print(f"tau={TAU}: {len(runs[TAU][0])} frames")

fig, axes = plt.subplots(1, len(TAUS), figsize=(5 * len(TAUS), 5.4), dpi=70)
ims = []
for ax, TAU in zip(axes, TAUS):
    im = ax.imshow(runs[TAU][0][0], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(rf"$\tau_Q = {TAU}$", fontsize=18)
    ax.set_xticks([]); ax.set_yticks([])
    ims.append(im)
title = fig.suptitle("", fontsize=20)

def update(k):
    for im, TAU in zip(ims, TAUS):
        f = runs[TAU][0][k]
        v = 2.5 * f.std() + 1e-30          # rescale each frame: field grows by orders of magnitude
        im.set_data(f); im.set_clim(-v, v)
    t_frac = runs[TAUS[0]][1][k] / TAUS[0]
    title.set_text(rf"$t/\tau_Q = {t_frac:+.2f}$")
    return ims + [title]

anim = animation.FuncAnimation(fig, update, frames=NFRAMES, interval=1000 / FPS)
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
anim.save(OUTFILE, writer="pillow", fps=FPS)
plt.close(fig)
print("saved:", OUTFILE, f"({os.path.getsize(OUTFILE) / 1e6:.1f} MB)")