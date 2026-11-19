"""Generate 1D KZ quench data for three environments, one chunk per call.

Same lattice, potential, ramp and snapshot grid as scripts/kz1d/generate_data.py (Suzuki).
Only the environment changes:

    classical : phi_tt + eta phi_t - phi_xx + V' = zeta,  white noise 2 eta theta,   phi = pi = 0 at t_i
    qlangevin : same, but zeta has the quantum (Caldeira-Leggett) spectrum
                S(w) = eta hbar w coth(hbar w / 2T) / (1 + w^2/Omega_c^2),          phi = pi = 0 at t_i
    twa       : phi_tt - phi_xx + V' = 0 (closed),  Wigner vacuum at eps_i

Example usage (from the repo root):
    python -m scripts.kz1d.generate_open twa 128 0          # case, tau, chunk index
    python -m scripts.kz1d.generate_open qlangevin 128 3 --n_chunks 10

Output: data/1D_open/{case}/tau{TAU}/chunk_{chunk:03d}.npz
    phi_hist  (n, n_snap, DOF) float32   snapshots at t = ts
    ts        (n_snap,)                  snapshot times
    phi_final (n, DOF)        float32    t = TAU (eps = 1)
    n_defects (n,)            int
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, time
import numpy as np
import src.kz as kz

# ---- parameters (Suzuki, unchanged) ----
kz.DOF, kz.dx      = 1024, 0.5
kz.EPS_I, kz.EPS_F = -1.5, 1.0

CASES     = ["classical", "qlangevin", "twa"]
N_SAMPLES = 3000                 # total per (case, tau)
N_CHUNKS  = 10                   # array tasks per (case, tau)
DT        = 0.05                 # RK4 step
SNAP_DT   = 1.0                  # snapshot spacing
ETA       = 1.0                  # damping (classical, qlangevin)
THETA     = 1e-8                 # Suzuki temperature (classical)
HBAR      = 2*THETA/np.sqrt(-kz.EPS_I/2)   # matches the k=0 thermal seed, ~2.31e-8 (qlangevin, twa)
T_BATH    = 0.0                  # qlangevin bath temperature (0 = pure zero-point noise)
OMEGA_C   = 4.0                  # qlangevin Drude cutoff
BATCH     = {"classical": 100, "qlangevin": 10, "twa": 100}
SEED0     = 990304
OUTDIR    = "data/1D_open"


def generate(case, TAU, chunk, a):
    t0_wall = time.time()
    kz.TAU = float(TAU)
    n_per  = a.n_samples // a.n_chunks
    rng    = np.random.default_rng([a.seed0, CASES.index(case), int(TAU), chunk])

    n_steps = int(round((kz.EPS_F - kz.EPS_I)*kz.TAU/a.dt))
    t_hist, dt_actual = kz.make_time_grid(n_steps)
    stride = max(1, int(round(SNAP_DT/dt_actual)))
    ts     = t_hist[::stride]

    # TWA: Wigner vacuum of the quadratic part at eps_i
    k     = 2*np.pi*np.fft.fftfreq(kz.DOF, d=kz.dx)
    w_k   = np.sqrt((4/kz.dx**2)*np.sin(k*kz.dx/2)**2 - kz.EPS_I/2)
    a_phi = np.sqrt(a.hbar/(2*kz.dx*w_k))
    a_pi  = np.sqrt(a.hbar*w_k/(2*kz.dx))

    # qlangevin: bath spectrum -> filter on white noise, <zeta^2> = int S dw/2pi / dx
    omega = 2*np.pi*np.fft.rfftfreq(n_steps, d=dt_actual)
    if a.T_bath > 0:
        S = np.empty_like(omega)
        S[0]  = 2*a.eta*a.T_bath
        S[1:] = a.eta*a.hbar*omega[1:]/np.tanh(a.hbar*omega[1:]/(2*a.T_bath))
    else:
        S = a.eta*a.hbar*np.abs(omega)
    S    = S/(1 + omega**2/a.omega_c**2)
    filt = np.sqrt(S/(dt_actual*kz.dx))[:, None, None]

    phi_hist = np.empty((n_per, len(ts), kz.DOF), dtype=np.float32)
    batch    = BATCH[case]
    for b0 in range(0, n_per, batch):
        nb = min(batch, n_per - b0)

        if case == "classical":
            kz.eta, kz.theta = a.eta, a.theta
            ph, _ = kz.run(t_hist, dt_actual, n_real=nb, seed=int(rng.integers(2**31)), stride=stride)
            phi_hist[b0:b0+nb] = np.transpose(ph, (1, 0, 2))

        elif case == "twa":
            kz.eta, kz.theta = 0.0, 0.0
            phi0 = np.real(np.fft.ifft(a_phi*np.fft.fft(rng.standard_normal((nb, kz.DOF)), axis=1), axis=1))
            pi0  = np.real(np.fft.ifft(a_pi *np.fft.fft(rng.standard_normal((nb, kz.DOF)), axis=1), axis=1))
            ph, _ = kz.run(t_hist, dt_actual, n_real=nb, stride=stride, phi0=phi0, pi0=pi0)
            phi_hist[b0:b0+nb] = np.transpose(ph, (1, 0, 2))

        elif case == "qlangevin":
            kz.eta, kz.theta = a.eta, 0.0        # noise added by hand below
            zeta = np.fft.irfft(filt*np.fft.rfft(rng.standard_normal((n_steps, nb, kz.DOF)), axis=0),
                                n=n_steps, axis=0)
            phi, pi = np.zeros((nb, kz.DOF)), np.zeros((nb, kz.DOF))
            phi_hist[b0:b0+nb, 0] = phi
            j = 1
            for n, t in enumerate(t_hist[:-1]):
                phi, pi = kz.rk4_step(phi, pi, t, dt_actual, t_hist[0], t_hist[-1])
                pi += dt_actual*zeta[n]
                if (n + 1) % stride == 0 and j < len(ts):
                    phi_hist[b0:b0+nb, j] = phi
                    j += 1
            del zeta

        el = time.time() - t0_wall
        print(f"  [{case} tau={TAU} chunk={chunk}] {b0+nb}/{n_per}  ({el:.0f}s)", flush=True)

    phi_final = phi_hist[:, -1, :]
    n_def = kz.count_defects(phi_final.astype(np.float64))
    outdir = os.path.join(a.outdir, case, f"tau{int(TAU)}")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"chunk_{chunk:03d}.npz")
    np.savez(path, phi_hist=phi_hist, ts=ts, phi_final=phi_final, n_defects=n_def,
             case=case, TAU=TAU, dt=dt_actual, dx=kz.dx, DOF=kz.DOF, eta=kz.eta,
             theta=a.theta if case == "classical" else 0.0,
             hbar=a.hbar if case != "classical" else 0.0,
             T_bath=a.T_bath, omega_c=a.omega_c, seed0=a.seed0, chunk=chunk)
    print(f"{case} tau={TAU} chunk={chunk}: defects mean={n_def.mean():.1f} sd={n_def.std():.1f}  "
          f"({time.time() - t0_wall:.0f}s) -> {path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=CASES)
    p.add_argument("tau", type=float)
    p.add_argument("chunk", type=int)
    p.add_argument("--n_samples", type=int, default=N_SAMPLES)
    p.add_argument("--n_chunks", type=int, default=N_CHUNKS)
    p.add_argument("--dt", type=float, default=DT)
    p.add_argument("--eta", type=float, default=ETA)
    p.add_argument("--theta", type=float, default=THETA)
    p.add_argument("--hbar", type=float, default=HBAR)
    p.add_argument("--T_bath", type=float, default=T_BATH)
    p.add_argument("--omega_c", type=float, default=OMEGA_C)
    p.add_argument("--seed0", type=int, default=SEED0)
    p.add_argument("--outdir", default=OUTDIR)
    a = p.parse_args()
    generate(a.case, a.tau, a.chunk, a)
