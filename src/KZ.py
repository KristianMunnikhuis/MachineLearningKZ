"""1+1D Landau-Ginzburg Langevin solver.

    phi_tt + eta*phi_t - phi_xx + dV/dphi = noise
    V(phi) = (phi^4 - 2*eps*phi^2)/8
    <noise(x,t) noise(x',t')> = 2*eta*theta*delta(x-x')*delta(t-t')

eps ramps linearly from EPS_I to EPS_F over the time grid you pass to run().

Usage:
    import kz
    kz.TAU = 256.0
    t_hist, dt = kz.make_time_grid()
    phi_hist, pi_hist = kz.run(t_hist, dt, seed=0)
    kz.heatmap(phi_hist, t_hist)
"""
import numpy as np
import matplotlib.pyplot as plt

# ---- parameters ----
DOF   = 1024      # grid points
dx    = 0.5       # lattice spacing
eta   = 1.0       # damping
theta = 1e-8      # temperature (noise strength)
TAU   = 128.0     # quench timescale
EPS_I = -1.5      # eps at the start of the run
EPS_F = 1.0       # eps at the end of the run

L    = lambda: DOF*dx
grid = lambda: np.arange(DOF)*dx


def make_time_grid(n_steps=6400):
    """Time array spanning the eps ramp at rate 1/TAU. Returns (t_hist, dt)."""
    return np.linspace(EPS_I*TAU, EPS_F*TAU, n_steps+1, retstep=True)


def epsilon(t, t0, t1):
    """Linear ramp EPS_I -> EPS_F as t goes t0 -> t1."""
    s = (t - t0)/(t1 - t0)
    return EPS_I + (EPS_F - EPS_I)*s


# ---- model ----
def laplacian(phi):
    return (np.roll(phi, 1, axis=-1) - 2*phi + np.roll(phi, -1, axis=-1))/dx/dx

def partial_V(phi, eps):
    return 0.5*(phi**3 - eps*phi)

def deriv(phi, pi, eps):
    return pi, -eta*pi + laplacian(phi) - partial_V(phi, eps)

def rk4_step(phi, pi, t, dt, t0, t1):
    e0 = epsilon(t,        t0, t1)
    eh = epsilon(t + dt/2, t0, t1)
    e1 = epsilon(t + dt,   t0, t1)
    a1, b1 = deriv(phi,           pi,           e0)
    a2, b2 = deriv(phi + dt/2*a1, pi + dt/2*b1, eh)
    a3, b3 = deriv(phi + dt/2*a2, pi + dt/2*b2, eh)
    a4, b4 = deriv(phi + dt*a3,   pi + dt*b3,   e1)
    return (phi + dt/6*(a1 + 2*a2 + 2*a3 + a4),
            pi  + dt/6*(b1 + 2*b2 + 2*b3 + b4))


# ---- driver ----
def run(t_hist, dt, n_real=None, seed=None, stride=1, phi0=None, pi0=None):
    """Integrate over t_hist. Returns (phi_hist, pi_hist).

    n_real : None -> shape (n_saved, DOF);  int -> (n_saved, n_real, DOF)
    phi0   : scalar or array initial condition (default 0)
    stride : store every stride-th step
    """
    rng = np.random.default_rng(seed)
    shape = (DOF,) if n_real is None else (n_real, DOF)
    t0, t1 = t_hist[0], t_hist[-1]

    phi = np.zeros(shape) if phi0 is None else np.broadcast_to(np.asarray(phi0, float), shape).copy()
    pi  = np.zeros(shape) if pi0  is None else np.broadcast_to(np.asarray(pi0,  float), shape).copy()
    amp = np.sqrt(2*eta*theta*dt/dx)

    keep = np.arange(0, len(t_hist), stride)
    phi_hist = np.zeros((len(keep),) + shape)
    pi_hist  = np.zeros((len(keep),) + shape)
    phi_hist[0], pi_hist[0] = phi, pi

    j = 1
    for n, t in enumerate(t_hist[:-1]):
        phi, pi = rk4_step(phi, pi, t, dt, t0, t1)
        if theta > 0:
            pi += amp*rng.standard_normal(shape)
        if (n+1) % stride == 0 and j < len(keep):
            phi_hist[j], pi_hist[j] = phi, pi
            j += 1

    return phi_hist, pi_hist


# ---- initial conditions ----
def ic_uniform(eps):
    """Single domain sitting in the +sqrt(eps) well."""
    return np.sqrt(max(eps, 0.0))*np.ones(DOF)

def ic_domains(eps, n_kinks=8):
    """n_kinks equal-width domains alternating +/- sqrt(eps). n_kinks must be even."""
    return np.sqrt(max(eps, 0.0))*np.sign(np.sin(n_kinks*np.pi*grid()/L() + 1e-12))


# ---- diagnostics ----
def count_defects(phi):
    s = np.sign(phi)
    return np.sum(s != np.roll(s, -1, axis=-1), axis=-1)

def defect_positions(phi):
    p, q = phi, np.roll(phi, -1)
    i = np.nonzero(np.sign(p) != np.sign(q))[0]
    return (i + p[i]/(p[i] - q[i]))*dx

def t_hat():
    """Overdamped KZ freeze-out time."""
    return np.sqrt(2*eta*TAU)

def equilibrium_rms(eps):
    """phi_rms in equilibrium at fixed eps: sqrt(theta/2m), m^2 = |eps|/2 (sym) or eps (broken)."""
    m = np.sqrt(eps) if eps > 0 else np.sqrt(abs(eps)/2)
    return np.sqrt(theta/(2*m))


def heatmap(hist, t_hist, log=False, ax=None, cmap=None, label=r'$\phi$'):
    """Space-time heatmap. log=True plots log10|field| with a sequential map."""
    if ax is None:
        _, ax = plt.subplots()
    ts = np.linspace(t_hist[0], t_hist[-1], len(hist))
    if log:
        img = ax.imshow(np.log10(np.abs(hist) + 1e-300), aspect='auto', origin='lower',
                        extent=[0, L(), ts[0], ts[-1]], cmap=cmap or 'viridis')
        label = r'$\log_{10}|$' + label.strip('$') + r'$|$'
    else:
        M = np.abs(hist).max()
        img = ax.imshow(hist, aspect='auto', origin='lower',
                        extent=[0, L(), ts[0], ts[-1]],
                        cmap=cmap or 'RdBu_r', vmin=-M, vmax=M)
    plt.colorbar(img, ax=ax, label=label)
    ax.set_xlabel('x'); ax.set_ylabel('t')
    return ax