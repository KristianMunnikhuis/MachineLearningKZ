"""Analysis helpers for the 2D KZ predictability experiments.

Import in a notebook (from notebooks/):
    import sys; sys.path.insert(0, "..")
    import src.analysis as an

    res = an.load_results(32)
    an.plot_error_curve(res, tau=32)
"""
import glob
import json
import os
import numpy as np
import torch
import src.kz_ml as kzml

ROOT    = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results", "v2")
DATA    = os.path.join(ROOT, "data", "2D_v2")
# ---------------------------------------------------------------- results


def load_results(tau, target="y_end", results_dir=RESULTS):
    """All finished runs for one tau and target, sorted by input time (units of t_hat).

    Returns a list of dicts (the JSON contents, plus 'tag' and 'has_ckpt').
    """
    out = []
    for f in glob.glob(os.path.join(results_dir, f"tau{int(tau)}_that*_{target}.json")):
        r = json.load(open(f))
        r["tag"] = f[:-5]
        r["has_ckpt"] = os.path.exists(f[:-5] + ".pt")
        out.append(r)
    out.sort(key=lambda r: r["t_used"])
    return out


def as_arrays(res):
    """Pull (t, baseline_error, model_error) out of a results list."""
    t    = np.array([r["t_used"]       for r in res])
    eb   = np.array([1 - r["baseline"] for r in res])
    em   = np.array([1 - r["test_acc"] for r in res])
    return t, eb, em


def t_star(res, threshold=0.25):
    """Input time (units of t_hat) at which the model error first drops below `threshold`.

    Linear interpolation between the bracketing points. Returns np.nan if the curve never crosses.
    """
    t, _, em = as_arrays(res)
    for i in range(len(em) - 1):
        if em[i] > threshold >= em[i+1]:
            f = (em[i] - threshold) / (em[i] - em[i+1])
            return t[i] + f * (t[i+1] - t[i])
    return np.nan



# ---------------------------------------------------------------- models

def load_model(tag):
    """Load a trained U-Net and its metadata from a results tag."""
    r = json.load(open(tag + ".json"))
    model = kzml.UNet()
    model.load_state_dict(torch.load(tag + ".pt", map_location="cpu"))
    model.eval()
    return model, r


def predict(model, X, scale):
    """Binary sign-map predictions for a stack of input fields (n, N, N)."""
    with torch.no_grad():
        logits = model(torch.from_numpy(X[:, None].astype(np.float32) / scale))
    return (logits > 0).numpy()[:, 0]


def load_chunk(tau, chunk=0, data_dir=DATA):
    """One chunk of v2 data for a given tau."""
    return np.load(os.path.join(data_dir, f"tau{int(tau)}", f"chunk_{chunk:03d}.npz"))


def snapshot_at(d, t_hat_in):
    """The snapshot nearest t_hat_in (units of t_hat), as float32, plus the time used."""
    i = int(np.argmin(np.abs(d["snap_hat"] - t_hat_in)))
    return d["snaps"][:, i].astype(np.float32), float(d["snap_hat"][i])

# ---------------------------------------------------------------- physics

def wall_length(s, dx=0.5):
    """Domain-wall length of a boolean or sign field, last two axes."""
    b = np.asarray(s) > 0
    bx = (b != np.roll(b, -1, axis=-1)).sum(axis=(-1, -2))
    by = (b != np.roll(b, -1, axis=-2)).sum(axis=(-1, -2))
    return (bx + by) * dx


def correlation_length(field, dx=0.5):
    """Rough domain size from the first zero crossing of the radial
    correlation function of sign(field). Returns one value per sample."""
    s = np.where(np.asarray(field) > 0, 1.0, -1.0)
    n = s.shape[-1]

    # radially averaged correlation via FFT
    f = np.fft.fft2(s, axes=(-2, -1))
    c = np.fft.ifft2(f * np.conj(f), axes=(-2, -1)).real / (n * n)

    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n,
                         indexing="ij")
    r = np.round(np.sqrt(kx**2 + ky**2)).astype(int)
    rmax = n // 2

    out = np.zeros(len(s))
    for i in range(len(s)):
        prof = np.array([c[i][r == k].mean() for k in range(rmax)])
        prof = prof / prof[0]
        below = np.where(prof < 0)[0]
        out[i] = below[0] * dx if len(below) else rmax * dx
    return out


def error_breakdown(pred, true, width=2):
    """Split errors into 'near a true wall' and 'bulk' (whole-domain) errors.

    Returns (fraction_near_wall, fraction_bulk) of all wrong pixels.
    """
    from scipy.ndimage import binary_dilation

    t = np.asarray(true) > 0
    wrong = (np.asarray(pred) > 0) != t

    # true wall sites: differ from a neighbour
    wall = (t != np.roll(t, -1, axis=-1)) | (t != np.roll(t, -1, axis=-2))

    near, bulk = [], []
    for i in range(len(t)):
        band = binary_dilation(wall[i], iterations=width)
        w = wrong[i]
        total = w.sum()
        if total == 0:
            near.append(np.nan); bulk.append(np.nan); continue
        near.append((w & band).sum() / total)
        bulk.append((w & ~band).sum() / total)
    return np.array(near), np.array(bulk)
