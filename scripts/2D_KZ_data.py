"""2D data generation with resume.

Writes one shard per chunk of samples, so an interrupted run picks up where it
left off: rerun with the same settings and only the missing shards are computed.
Seeds are deterministic per sample (SEED0 + index), so resumed data is identical
to data from an uninterrupted run.

Only the requested input windows and the final configuration are stored -- the
full history is far too large in 2D.

    python generate_data_2d.py                    # defaults
    python generate_data_2d.py --n 200 --N 128    # smaller test

Load:
    import generate_data_2d as g2
    d = g2.load(tau=128, N=128)
    X = d['win_50']        # (n, 11, N, N)  input window at t = 50
    Y = d['phi_final']     # (n, N, N)
"""
import argparse, os, time, glob
import numpy as np
import KZ_2D as kz2

# ---- parameters ----
kz2.dx, kz2.eta, kz2.theta = 0.5, 1.0, 1e-8
kz2.EPS_I, kz2.EPS_F       = -1.5, 1.0

N_GRID    = 128            # sites per side
TAU       = 128.0
N_SAMPLES = 3000
N_STEPS   = 3200           # dt = 0.1 at TAU=128; checked against dt=0.05
T_SAMPLE  = [20, 50, 70]   # window centres to store
WIN       = 5              # window is t-WIN .. t+WIN, step SNAP_DT
SNAP_DT   = 1.0
CHUNK     = 50             # samples per shard
SEED0     = 0
OUTDIR    = 'data2d'


def _tag(tau, N):
    return f'tau{int(tau)}_N{int(N)}'


def _one_chunk(i0, n, tau, N, n_steps, t_sample, win, seed0):
    """Integrate n realizations, return (windows dict, final) as float32."""
    kz2.N, kz2.TAU = int(N), float(tau)
    t_hist, dt = kz2.make_time_grid(n_steps=n_steps)
    stride = max(1, int(round(SNAP_DT/dt)))
    t0, t1 = t_hist[0], t_hist[-1]

    # step indices we actually need to keep
    want = {}
    for tc in t_sample:
        for k in range(-win, win+1):
            j = int(round((tc + k*SNAP_DT - t0)/dt))
            want.setdefault(j, []).append((tc, k+win))
    j_final = len(t_hist) - 1

    shape = (n, N, N)
    rng = np.random.default_rng([seed0, i0])
    phi = np.zeros(shape)
    pi  = np.zeros(shape)
    amp = np.sqrt(2*kz2.eta*kz2.theta*dt/kz2.dx**2)

    wins = {tc: np.zeros((n, 2*win+1, N, N), dtype=np.float32) for tc in t_sample}

    for s, t in enumerate(t_hist[:-1]):
        phi, pi = kz2.rk4_step(phi, pi, t, dt, t0, t1)
        if kz2.theta > 0:
            pi += amp*rng.standard_normal(shape)
        step = s + 1
        if step in want:
            snap = phi.astype(np.float32)
            for tc, k in want[step]:
                wins[tc][:, k] = snap

    return wins, phi.astype(np.float32), stride, dt


def generate(tau=TAU, N=N_GRID, n_samples=N_SAMPLES, n_steps=N_STEPS,
             t_sample=T_SAMPLE, win=WIN, chunk=CHUNK, seed0=SEED0,
             outdir=OUTDIR, verbose=True):
    tag = _tag(tau, N)
    d   = os.path.join(outdir, tag)
    os.makedirs(d, exist_ok=True)

    snap_mb = N*N*4/1e6
    total_gb = n_samples*(len(t_sample)*(2*win+1) + 1)*snap_mb/1000
    if verbose:
        kz2.N, kz2.TAU = int(N), float(tau)
        print(f'{tag}: {n_samples} samples, t_sample={t_sample}, '
              f'xi_hat={kz2.xi_hat():.1f}, L={kz2.L():.0f}, '
              f'est. {total_gb:.1f} GB', flush=True)

    starts  = list(range(0, n_samples, chunk))
    todo    = [i0 for i0 in starts
               if not os.path.exists(os.path.join(d, f'chunk_{i0:06d}.npz'))]
    if verbose:
        print(f'  {len(starts)-len(todo)}/{len(starts)} shards already present',
              flush=True)

    t_start = time.time()
    for m, i0 in enumerate(todo):
        n = min(chunk, n_samples - i0)
        wins, final, stride, dt = _one_chunk(i0, n, tau, N, n_steps,
                                             t_sample, win, seed0)
        payload = {f'win_{tc}': wins[tc] for tc in t_sample}
        payload.update(phi_final=final,
                       wall_len=kz2.wall_length(final.astype(np.float64)),
                       i0=i0, n=n, tau=tau, N=N, dt=dt, stride=stride,
                       t_sample=np.array(t_sample), win=win, seed0=seed0)
        tmp = os.path.join(d, f'chunk_{i0:06d}.tmp.npz')
        np.savez(tmp, **payload)                      # write then rename:
        os.replace(tmp, os.path.join(d, f'chunk_{i0:06d}.npz'))   # never a partial shard
        if verbose:
            el = time.time() - t_start
            print(f'  chunk {i0:6d}  ({m+1}/{len(todo)})  '
                  f'{el:.0f}s elapsed, {el/(m+1)*(len(todo)-m-1):.0f}s left',
                  flush=True)
    return d


def load(tau=TAU, N=N_GRID, outdir=OUTDIR, n_max=None):
    """Concatenate all shards. Returns a dict of arrays."""
    d = os.path.join(outdir, _tag(tau, N))
    files = sorted(glob.glob(os.path.join(d, 'chunk_*.npz')))
    if not files:
        raise FileNotFoundError(f'no shards in {d}')
    parts = [np.load(f) for f in files]
    t_sample = list(parts[0]['t_sample'])
    out = {f'win_{tc}': np.concatenate([p[f'win_{tc}'] for p in parts])
           for tc in t_sample}
    out['phi_final'] = np.concatenate([p['phi_final'] for p in parts])
    out['wall_len']  = np.concatenate([p['wall_len']  for p in parts])
    for k in ['tau', 'N', 'dt', 'stride', 'win', 'seed0']:
        out[k] = parts[0][k]
    out['t_sample'] = t_sample
    if n_max:
        for k in list(out):
            if isinstance(out[k], np.ndarray) and out[k].ndim >= 1 and len(out[k]) > n_max:
                out[k] = out[k][:n_max]
    return out


def progress(tau=TAU, N=N_GRID, n_samples=N_SAMPLES, chunk=CHUNK, outdir=OUTDIR):
    d = os.path.join(outdir, _tag(tau, N))
    have = len(glob.glob(os.path.join(d, 'chunk_*.npz')))
    need = len(range(0, n_samples, chunk))
    print(f'{_tag(tau, N)}: {have}/{need} shards ({have*chunk} samples)')
    return have, need


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tau',   type=float, default=TAU)
    p.add_argument('--N',     type=int,   default=N_GRID)
    p.add_argument('--n',     type=int,   default=N_SAMPLES)
    p.add_argument('--steps', type=int,   default=N_STEPS)
    p.add_argument('--chunk', type=int,   default=CHUNK)
    a = p.parse_args()
    generate(tau=a.tau, N=a.N, n_samples=a.n, n_steps=a.steps, chunk=a.chunk)