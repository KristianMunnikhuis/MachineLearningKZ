"""Figure 2: dynamics of the order parameter, <|phi(x,t)|> vs t.
Paper: arXiv:2508.20347, Fig. 2.  tau_Q = 128.
<.> is the spatial average over x, per the paper's caption.
"""
import numpy as np
import matplotlib.pyplot as plt
import KZ as kz

# ---- parameters ----
kz.DOF, kz.dx      = 1024, 0.5
kz.eta, kz.theta   = 1.0, 1e-8
kz.EPS_I, kz.EPS_F = -1.5, 1.0
kz.TAU             = 128.0

DT     = 0.05
SEED   = 0
N_REAL = None        # None = single realization; int = also show ensemble mean

# ---- data ----
n_steps = int(round((kz.EPS_F - kz.EPS_I)*kz.TAU/DT))
t_hist, dt = kz.make_time_grid(n_steps)
stride = max(1, int(round(0.5/dt)))

phi, pi = kz.run(t_hist, dt, n_real=N_REAL, seed=SEED, stride=stride)

ts   = t_hist[::stride][:len(phi)]
eps  = ts/kz.TAU
mphi = np.abs(phi).mean(axis=-1)        # spatial average
mpi  = np.abs(pi ).mean(axis=-1)
if N_REAL is not None:                  # then average over realizations too
    mphi = mphi.mean(axis=-1)
    mpi  = mpi.mean(axis=-1)

post   = ts > 0
t_red  = ts[post][int(np.argmax(mpi[post]))]
t_hat  = np.sqrt(2*kz.eta*kz.TAU)
sqeps  = np.sqrt(np.clip(eps, 0, None))

# ---- plot ----
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(ts, mphi, 'k-', lw=1.5, label=r'$\langle|\phi(x,t)|\rangle$')
ax.plot(ts, sqeps, '--', color='grey', lw=1.2, label=r'$\sqrt{\epsilon}$')
ax.axvline(t_red, color='r', lw=1.2,
           label=rf'max $\langle|\dot\phi|\rangle$:  $t={t_red:.0f}$')
#ax.axvline(t_hat, color='b', ls=':', lw=1.2,
    #       label=rf'$\hat t=\sqrt{{2\eta\tau_Q}}={t_hat:.0f}$')
ax.axvline(0, color='k', lw=0.5, alpha=0.4)

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$\langle|\phi(x,t)|\rangle$')
ax.set_xlim(ts[0], ts[-1])
ax.set_title(rf'$\tau_Q={kz.TAU:.0f}$')
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(-50,ts[-1])
plt.tight_layout()
plt.savefig('figure2.png', dpi=150)
plt.show()