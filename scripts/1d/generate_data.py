"""Generate and store the training data for the 1d model.

For each tau_Q, runs N_SAMPLES independent noise realizations and stores the
snapshot history at integer tiemsteps and saves the final configuration.

Example usage:

    python generate_data.py                 # all tau_Q, paper defaults
    python generate_data.py 128             # one tau_Q
    python generate_data.py 128 --n 200     # fewer samples (testing)

Output: data/kz_tau{TAU}.npz
    phi_hist  (N, n_snap, DOF) float32   snapshots at t = ts
    ts        (n_snap,)        float64   snapshot times
    phi_final (N, DOF)         float32   target, t = TAU (eps = 1)
    n_defects (N,)             int
"""
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import argparse, os, time
import numpy as np
import src.kz as kz

# ---- parameters ----
kz.DOF, kz.dx      = 1024, 0.5
kz.eta, kz.theta   = 1.0, 1e-8
kz.EPS_I, kz.EPS_F = -1.5, 1.0

TAUS      = [128.0, 256.0, 512.0]
N_SAMPLES = 3000
DT        = 0.05      # RK4 step size
SNAP_DT   = 1.0       # snapshot spacing
BATCH     = 100       # realizations per pass
SEED0     = 990304    # Initial Seed
OUTDIR    = 'data'

# only snapshots in this window are kept, to save space.
# covers every sampling time in Fig. 4 (min -195) plus a +/-5 window.
T_KEEP_MIN = -220.0


def generate(TAU, n_samples=N_SAMPLES, dt=DT, batch=BATCH, seed0=SEED0,
             outdir=OUTDIR, verbose=True):
    #Set time
    t0 = time.time()

    #Set Tau
    kz.TAU = float(TAU)
    #calculate number of steps
    n_steps = int(round((kz.EPS_F - kz.EPS_I)*kz.TAU/dt))
    #Generaet time grid
    t_hist, dt_actual = kz.make_time_grid(n_steps)
    #Calculate stride
    stride = max(1, int(round(SNAP_DT/dt_actual)))
    #Pick times 
    ts_all = t_hist[::stride]
    keep   = ts_all >= T_KEEP_MIN
    ts     = ts_all[keep]
    #Optional Print 
    if verbose:
        print(f'tau_Q={TAU:.0f}  dt={dt_actual:.4f}  n_steps={n_steps}  '
              f'stride={stride}  snapshots kept={keep.sum()}/{len(ts_all)}')
    #Empty data array
    phi_hist = np.empty((n_samples, keep.sum(), kz.DOF), dtype=np.float32)

    for b0 in range(0, n_samples, batch):
        nb = min(batch, n_samples - b0) #Only matters for last batch
        #Generate snapshots for the run
        ph, _ = kz.run(t_hist, dt_actual, n_real=nb, seed=seed0 + b0,
                       stride=stride)
        # ph is (n_snap, nb, DOF) -> (nb, n_snap_kept, DOF)
        phi_hist[b0:b0+nb] = np.transpose(ph[keep[:len(ph)]], (1, 0, 2)).astype(np.float32)
        if verbose:
            el = time.time() - t0
            print(f'  {b0+nb}/{n_samples}  ({el:.0f}s elapsed, '
                  f'{el/(b0+nb)*(n_samples-b0-nb):.0f}s left)', flush=True)

    phi_final = phi_hist[:, -1, :]
    n_def = kz.count_defects(phi_final.astype(np.float64))
    #Save Data
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'kz_tau{int(TAU)}.npz')
    np.savez_compressed(path, phi_hist=phi_hist, ts=ts, phi_final=phi_final,
                        n_defects=n_def, TAU=TAU, dt=dt_actual, dx=kz.dx,
                        DOF=kz.DOF, eta=kz.eta, theta=kz.theta, seed0=seed0)
    if verbose:
        mb = os.path.getsize(path)/1e6
        print(f'  -> {path}  ({mb:.0f} MB)  '
              f'defects: mean={n_def.mean():.1f} sd={n_def.std():.1f} '
              f'range=[{n_def.min()},{n_def.max()}]')
    return path


def load(TAU, outdir=OUTDIR):
    return np.load(os.path.join(outdir, f'kz_tau{int(TAU)}.npz'))


def window(d, t_center, win=5):
    """Slice the (2*win+1, DOF) input windows for every sample.
    Returns (N, 2*win+1, DOF)."""
    ts = d['ts']
    j  = int(np.argmin(abs(ts - t_center)))
    k  = int(round(1.0/(ts[1] - ts[0])))
    idx = np.arange(j - win*k, j + win*k + 1, k)
    if idx[0] < 0 or idx[-1] >= len(ts):
        raise ValueError(f't={t_center} window runs off the stored range '
                         f'[{ts[0]:.0f}, {ts[-1]:.0f}]')
    return d['phi_hist'][:, idx, :]

#main loop
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('taus', nargs='*', type=float, default=TAUS)
    p.add_argument('--n', type=int, default=N_SAMPLES)
    p.add_argument('--dt', type=float, default=DT)
    p.add_argument('--batch', type=int, default=BATCH)
    a = p.parse_args()
    for TAU in a.taus:
        generate(TAU, n_samples=a.n, dt=a.dt, batch=a.batch)