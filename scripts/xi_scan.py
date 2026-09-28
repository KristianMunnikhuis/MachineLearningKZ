"""Quick xi scan with snapshots: 20 samples per tau at N=256.

Standalone -- does not touch kz_chunk_2d.py or any existing data.

Run from the repo root:
    python -m scripts.xi_scan
    python -m scripts.xi_scan --taus 20 40 80 --n 20 --N 256

Writes one file per tau:  data/xi_scan_N256/tau40.npz
containing the full snapshot history, the final field, and measured xi.
"""
import argparse
import os
import time

import numpy as np
import src.KZ_2D as kz2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--taus",  type=float, nargs="+",
                   default=[10, 20, 40, 80, 160])
    p.add_argument("--n",     type=int,   default=20, help="samples per tau")
    p.add_argument("--N",     type=int,   default=256, help="sites per side")
    p.add_argument("--nsnap", type=int,   default=31,
                   help="snapshots per run, spread over the fracs below")
    p.add_argument("--fmin",  type=float, default=-0.5, help="first t/tau")
    p.add_argument("--fmax",  type=float, default=1.0,  help="last t/tau")
    p.add_argument("--outdir", type=str,  default=None)
    return p.parse_args()


def run_one_tau(tau, n_samples, N, fracs):
    """Integrate n_samples realizations, keeping snapshots at t = frac*tau.

    Returns (snaps, snap_times, phi_final, dt).
    """
    kz2.dx, kz2.eta, kz2.theta = 0.5, 1.0, 1e-8
    kz2.EPS_I, kz2.EPS_F = -1.5, 1.0
    kz2.N, kz2.TAU = int(N), float(tau)

    n_steps = int(round(25 * tau))              # keeps dt = 0.1
    t_hist, dt = kz2.make_time_grid(n_steps=n_steps)
    t0, t1 = t_hist[0], t_hist[-1]

    snap_times = np.array([f * tau for f in fracs])
    snap_steps = [int(round((t - t0) / dt)) for t in snap_times]
    keep = {j: i for i, j in enumerate(snap_steps)
            if 1 <= j <= len(t_hist) - 1}
    if len(keep) < len(snap_steps):
        print(f"  note: {len(snap_steps)-len(keep)} snapshot times fall "
              f"outside the run and are skipped")

    rng = np.random.default_rng()
    shape = (n_samples, N, N)
    phi = np.zeros(shape)
    pi  = np.zeros(shape)
    amp = np.sqrt(2 * kz2.eta * kz2.theta * dt / kz2.dx**2)

    snaps = np.zeros((n_samples, len(snap_times), N, N), dtype=np.float32)

    for step in range(1, len(t_hist)):
        phi, pi = kz2.rk4_step(phi, pi, t_hist[step - 1], dt, t0, t1)
        pi += amp * rng.standard_normal(shape)
        if step in keep:
            snaps[:, keep[step]] = phi

    return snaps, snap_times, phi.astype(np.float32), dt


def xi(field, dx=1.0, target=np.exp(-1.0)):
    """1/e decay length of the correlation function of sign(field)."""
    s = np.where(np.asarray(field) > 0, 1.0, -1.0)
    s = s.reshape(-1, *s.shape[-2:])
    n = s.shape[-1]
    s = s - s.mean(axis=(-2, -1), keepdims=True)

    f = np.fft.fft2(s, axes=(-2, -1))
    c = np.fft.ifft2(np.abs(f)**2, axes=(-2, -1)).real / (n * n)

    ax = np.fft.fftfreq(n) * n
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    rbin = np.round(np.sqrt(xx**2 + yy**2)).astype(int)
    nb = n // 2
    counts = np.bincount(rbin.ravel(), minlength=nb + 1)[:nb]

    out = np.empty(len(s))
    for i in range(len(s)):
        prof = np.bincount(rbin.ravel(), weights=c[i].ravel(),
                           minlength=nb + 1)[:nb] / counts
        prof = prof / prof[0]
        below = np.where(prof < target)[0]
        if len(below) == 0 or below[0] == 0:
            out[i] = np.nan
            continue
        j = below[0]
        frac = (prof[j - 1] - target) / (prof[j - 1] - prof[j])
        out[i] = (j - 1 + frac) * dx
    return out


def wall_length(field, dx=1.0):
    b = np.asarray(field) > 0
    bx = (b != np.roll(b, -1, axis=-1)).sum(axis=(-1, -2))
    by = (b != np.roll(b, -1, axis=-2)).sum(axis=(-1, -2))
    return (bx + by) * dx


def main():
    a = parse_args()
    outdir = a.outdir or f"data/xi_scan_N{a.N}"
    os.makedirs(outdir, exist_ok=True)

    fracs = np.linspace(a.fmin, a.fmax, a.nsnap)
    mb = a.n * a.nsnap * a.N**2 * 4 / 1e6
    print(f"N={a.N}, {a.n} samples/tau, {a.nsnap} snapshots "
          f"({mb:.0f} MB per tau)\n")
    print(f"{'tau':>6} {'xi':>8} {'+/-':>6} {'nan':>5} {'L/xi':>6} "
          f"{'wall':>8} {'time':>7}")

    taus, xis = [], []
    for tau in a.taus:
        out = os.path.join(outdir, f"tau{tau:g}.npz")
        if os.path.exists(out):
            d = np.load(out)
            print(f"{tau:6g} {float(d['xi_mean']):8.2f} "
                  f"{'':6} {'':5} {a.N/float(d['xi_mean']):6.1f} "
                  f"{'':8} (cached)")
            taus.append(tau); xis.append(float(d["xi_mean"]))
            continue

        t0 = time.time()
        snaps, snap_times, final, dt = run_one_tau(tau, a.n, a.N, fracs)
        x = xi(final)
        w = wall_length(final)

        tmp = os.path.join(outdir, f"tmp_tau{tau:g}.npz")
        np.savez(tmp, snaps=snaps, snap_times=snap_times, phi_final=final,
                 xi=x, xi_mean=np.nanmean(x), wall_len=w,
                 tau=tau, N=a.N, dt=dt, fracs=fracs)
        os.replace(tmp, out)

        taus.append(tau); xis.append(np.nanmean(x))
        print(f"{tau:6g} {np.nanmean(x):8.2f} {np.nanstd(x):6.2f} "
              f"{np.isnan(x).sum():5d} {a.N/np.nanmean(x):6.1f} "
              f"{w.mean():8.0f} {time.time()-t0:6.0f}s", flush=True)

    taus, xis = np.array(taus, float), np.array(xis)
    good = np.isfinite(xis)
    if good.sum() >= 2:
        p, cov = np.polyfit(np.log(taus[good]), np.log(xis[good]), 1, cov=True)
        print(f"\nxi ~ tau^({p[0]:.3f} +/- {np.sqrt(cov[0,0]):.3f})")
        print("subrange fits:")
        for lo in range(max(0, good.sum() - 2)):
            q = np.polyfit(np.log(taus[good][lo:]), np.log(xis[good][lo:]), 1)
            print(f"  tau >= {taus[good][lo]:g}: {q[0]:+.3f}")
    print(f"\nfiles in {outdir}/")


if __name__ == "__main__":
    main()