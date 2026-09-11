"""2+1D Landau-Ginzburg Langevin solver.

    phi_tt + eta*phi_t - lap(phi) + dV/dphi = noise
    V(phi) = (phi^4 - 2*eps*phi^2)/8
    <noise(r,t) noise(r',t')> = 2*eta*theta*delta^2(r-r')*delta(t-t')

eps ramps linearly from EPS_I to EPS_F over the time grid you pass to run().
Data generation only -- no plotting.

Usage:
    import KZ_2D as kz2
    kz2.N = 256
    t_hist, dt = kz2.make_time_grid()
    phi_hist, pi_hist = kz2.run(t_hist, dt, seed=0, stride=20)
    # phi_hist is (n_saved, N, N)
"""
import numpy as np

# ---- parameters ----
N     = 256       # grid points per side (total N*N)
dx    = 0.5       # lattice spacing
eta   = 1.0       # damping
theta = 1e-8      # temperature (noise strength)
TAU   = 128.0     # quench timescale
EPS_I = -1.5      # eps at the start of the run
EPS_F = 1.0       # eps at the end of the run

L    = lambda: N*dx
grid = lambda: np.arange(N)*dx

# RK4 stability, 2D 5-point stencil: max|omega| = 2*sqrt(2)/dx, dt*omega < 2.8
def dt_max():
    return dx


def make_time_grid(n_steps=6400):
    """Time array spanning the eps ramp at rate 1/TAU. Returns (t_hist, dt)."""
    return np.linspace(EPS_I*TAU, EPS_F*TAU, n_steps+1, retstep=True)


def epsilon(t, t0, t1):
    """Linear ramp EPS_I -> EPS_F as t goes t0 -> t1."""
    s = (t - t0)/(t1 - t0)
    return EPS_I + (EPS_F - EPS_I)*s


# ---- model ----
def laplacian(phi):
    """5-point stencil, periodic in both directions. Acts on the last two axes."""
    return (np.roll(phi,  1, axis=-1) + np.roll(phi, -1, axis=-1) +
            np.roll(phi,  1, axis=-2) + np.roll(phi, -1, axis=-2) -
            4*phi)/dx/dx

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
def run(t_hist, dt, n_real=None, seed=None, stride=1, phi0=None, pi0=None,
        check=True):
    """Integrate over t_hist. Returns (phi_hist, pi_hist).

    n_real : None -> shape (n_saved, N, N);  int -> (n_saved, n_real, N, N)
    phi0   : scalar or array initial condition (default 0)
    stride : store every stride-th step
    """
    if check and dt > dt_max():
        raise ValueError(f"dt={dt:.4g} exceeds RK4 stability limit dx={dt_max():.4g}")

    rng = np.random.default_rng(seed)
    shape = (N, N) if n_real is None else (n_real, N, N)
    t0, t1 = t_hist[0], t_hist[-1]

    phi = np.zeros(shape) if phi0 is None else np.broadcast_to(np.asarray(phi0, float), shape).copy()
    pi  = np.zeros(shape) if pi0  is None else np.broadcast_to(np.asarray(pi0,  float), shape).copy()
    amp = np.sqrt(2*eta*theta*dt/dx**2)          # note dx^2 in 2D

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

    if check and not np.isfinite(phi).all():
        raise RuntimeError("integration produced non-finite values; reduce dt")
    return phi_hist, pi_hist


# ---- initial conditions ----
def ic_uniform(eps):
    """Single domain sitting in the +sqrt(eps) well."""
    return np.sqrt(max(eps, 0.0))*np.ones((N, N))

def ic_stripes(eps, n_walls=8):
    """n_walls stripe domains alternating +/- sqrt(eps) along x. n_walls even."""
    return np.sqrt(max(eps, 0.0))*np.sign(
        np.sin(n_walls*np.pi*grid()/L() + 1e-12))[None, :]*np.ones((N, 1))


# ---- diagnostics ----
def wall_length(phi):
    """Total domain-wall length: bonds across which sign(phi) flips, times dx.
    Acts on the last two axes; works on a stack of snapshots."""
    s = np.sign(phi)
    bx = (s != np.roll(s, -1, axis=-1)).sum(axis=(-1, -2))
    by = (s != np.roll(s, -1, axis=-2)).sum(axis=(-1, -2))
    return (bx + by)*dx

def wall_mask(phi):
    """Boolean array marking sites adjacent to a sign flip. Single snapshot."""
    s = np.sign(phi)
    return ((s != np.roll(s, -1, axis=-1)) | (s != np.roll(s, 1, axis=-1)) |
            (s != np.roll(s, -1, axis=-2)) | (s != np.roll(s, 1, axis=-2)))

def domain_fraction(phi):
    """Fraction of sites with phi > 0. Should sit near 0.5 by symmetry."""
    return (phi > 0).mean(axis=(-1, -2))

def t_hat():
    """Overdamped KZ freeze-out time."""
    return np.sqrt(2*eta*TAU)

def xi_hat():
    """Overdamped KZ freeze-out length."""
    return np.sqrt(2.0)*(TAU/(2*eta))**0.25

def equilibrium_rms(eps):
    """phi_rms in equilibrium at fixed eps, 2D: theta/(4*pi) * log(k_max^2/m^2 + 1)."""
    m2 = eps if eps > 0 else abs(eps)/2
    kmax2 = (np.pi/dx)**2
    return np.sqrt(theta/(4*np.pi)*np.log(1 + kmax2/m2))