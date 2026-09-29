"""KZ scaling check: domain size at formation vs quench time.

Formation = first time phi_rms reaches FRAC * sqrt(eps). The ramp runs to EPS_F = 3 so
every tau reaches formation (tiny noise delays formation to ~5-6 t_hat).
Writes results/kz_scaling.json.

    python -m scripts.kz2d.kz_scaling
"""
import json
import os
import numpy as np
import src.kz_2d as kz2

# ---- inputs ----
TAUS = [8, 10, 13, 16, 20, 25, 32, 40, 51, 64, 81, 102, 128, 161, 203, 256]
N       = 256
N_REAL  = 10
FRAC    = 0.5
DT      = 0.25
SEED    = 0
OUTFILE = "results/kz_scaling.json"

kz2.dx, kz2.eta, kz2.theta = 0.5, 1.0, 1e-8
kz2.EPS_I, kz2.EPS_F = -1.0, 3.0
kz2.N = N

out = []
for tau in TAUS:
    kz2.TAU = float(tau)
    n_steps = int(round((kz2.EPS_F - kz2.EPS_I) * tau / DT))
    t_hist, dt = kz2.make_time_grid(n_steps=n_steps)
    t0, t1 = t_hist[0], t_hist[-1]

    rng = np.random.default_rng(SEED)
    phi = np.zeros((N_REAL, N, N))
    pi  = np.zeros((N_REAL, N, N))
    amp = np.sqrt(2 * kz2.eta * kz2.theta * dt / kz2.dx**2)

    for step in range(1, len(t_hist)):
        phi, pi = kz2.rk4_step(phi, pi, t_hist[step - 1], dt, t0, t1)
        pi += amp * rng.standard_normal(phi.shape)
        t   = t_hist[step]
        eps = t / tau
        if eps > 0 and np.sqrt((phi**2).mean()) >= FRAC * np.sqrt(eps):
            break
    else:
        print(f"tau={tau}: never reached formation, skipped", flush=True)
        continue

    ell    = (N * kz2.dx)**2 / kz2.wall_length(phi).mean()
    t_hat  = np.sqrt(2 * kz2.eta * tau)
    xi_hat = np.sqrt(2) * (tau / (2 * kz2.eta))**0.25
    out.append(dict(tau=tau, t_f=float(t), t_hat=t_hat, ell=float(ell), xi_hat=xi_hat))
    print(f"tau={tau:4d}  t_f/t_hat={t / t_hat:4.2f}  ell={ell:6.2f}  ell/xi_hat={ell / xi_hat:4.2f}", flush=True)

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
with open(OUTFILE, "w") as f:
    json.dump(dict(N=N, n_real=N_REAL, frac=FRAC, dt=DT, results=out), f, indent=2)
print("saved:", OUTFILE)