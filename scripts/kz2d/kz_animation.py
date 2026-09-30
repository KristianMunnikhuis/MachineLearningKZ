"""Animated GIF of 2D KZ quenches at several quench times, side by side.

Each panel is one tau_Q. All panels show the same KZ stage, t/t_hat, with
t_hat = sqrt(2*eta*tau_Q), so differences in domain size reflect xi_hat ~ tau_Q^(1/4).
Color is rescaled every frame, so it shows the sign pattern (domains), not amplitude.
Same conventions as scripts/kz2d/generate_data.py.

Run from the repo root:
    python -m scripts.figures.kz_animation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import src.kz_2d as kz2

# ---- inputs ----
TAUS        = [8, 32,64, 128]    # quench times, one panel each
N           = 512             # sites per side
DT          = 0.25            # RK4 step
T_START_HAT = -1.0            # first frame, in units of t_hat
T_MAX_HAT   = 10.0            # last frame, in units of t_hat (same as the dataset)
NFRAMES     = 120
SEED        = 88
FPS         = 20
HOLD_SEC    = 3               # seconds to hold on the final state
DPI         = 55
OUTFILE     = "figures/2d/kz_quench.gif"

kz2.dx, kz2.eta, kz2.theta = 0.5, 1.0, 1e-8
kz2.N = N

frame_hat = np.linspace(T_START_HAT, T_MAX_HAT, NFRAMES)   # common frame times, units of t_hat

# ---- simulate: one run per tau, record the field at each frame time ----
# custom loop instead of kz2.run(): run() stores every stride-th step plus pi, too much memory at N=512
runs = {}
for TAU in TAUS:
    t_hat = np.sqrt(2 * kz2.eta * TAU)
    kz2.TAU   = float(TAU)
    kz2.EPS_I = -1.0
    kz2.EPS_F = T_MAX_HAT * t_hat / TAU                      # run ends at t = T_MAX_HAT * t_hat

    n_steps = int(round((kz2.EPS_F - kz2.EPS_I) * TAU / DT))
    t_hist, dt = kz2.make_time_grid(n_steps=n_steps)
    t0, t1 = t_hist[0], t_hist[-1]
    frame_steps = [int(round((s * t_hat - t0) / dt)) for s in frame_hat]
    step_to_frame = {j: k for k, j in enumerate(frame_steps)}

    rng = np.random.default_rng(SEED)
    phi = np.zeros((N, N))
    pi  = np.zeros((N, N))
    amp = np.sqrt(2 * kz2.eta * kz2.theta * dt / kz2.dx**2)

    frames = np.zeros((NFRAMES, N, N), dtype=np.float32)
    for step in range(1, len(t_hist)):
        phi, pi = kz2.rk4_step(phi, pi, t_hist[step - 1], dt, t0, t1)
        pi += amp * rng.standard_normal(phi.shape)
        if step in step_to_frame:
            frames[step_to_frame[step]] = phi

    runs[TAU] = frames
    print(f"tau={TAU}: t_hat={t_hat:.2f}, {n_steps} steps", flush=True)

# ---- animate ----
fig, axes = plt.subplots(1, len(TAUS), figsize=(5 * len(TAUS), 5.4), dpi=DPI)
ims = []
for ax, TAU in zip(axes, TAUS):
    im = ax.imshow(runs[TAU][0], cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(rf"$\tau_Q = {TAU}$", fontsize=18)
    ax.set_xticks([]); ax.set_yticks([])
    ims.append(im)
title = fig.suptitle("", fontsize=20)

def update(k):
    for im, TAU in zip(ims, TAUS):
        f = runs[TAU][k]
        v = 2.5 * f.std() + 1e-30          # rescale each frame: field grows by orders of magnitude
        im.set_data(f); im.set_clim(-v, v)
    title.set_text(rf"$t/\hat t = {frame_hat[k]:+.1f}$")
    return ims + [title]

# repeat the last frame to hold on the final state
frame_order = list(range(NFRAMES)) + [NFRAMES - 1] * int(HOLD_SEC * FPS)
anim = animation.FuncAnimation(fig, update, frames=frame_order, interval=1000 / FPS)

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
anim.save(OUTFILE, writer="pillow", fps=FPS)
plt.close(fig)
print("saved:", OUTFILE, f"({os.path.getsize(OUTFILE) / 1e6:.1f} MB)")