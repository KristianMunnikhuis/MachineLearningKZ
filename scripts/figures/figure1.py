"""Figure 1: input window (t = 55, 60, 65) and the final configuration.
Paper: arXiv:2508.20347, Fig. 1 / Fig. 3.  tau_Q = 128.
"""
import numpy as np
import matplotlib.pyplot as plt
import src.KZ as kz

# ---- parameters ----
kz.DOF, kz.dx      = 1024, 0.5
kz.eta, kz.theta   = 1.0, 1e-8
kz.EPS_I, kz.EPS_F = -1.5, 1.0
kz.TAU             = 128.0

DT      = 0.05
SEED    = 0
T_WIN   = [55, 60, 65]

# ---- data ----
n_steps = int(round((kz.EPS_F - kz.EPS_I)*kz.TAU/DT))
t_hist, dt = kz.make_time_grid(n_steps)
stride = max(1, int(round(1.0/dt)))

phi, pi = kz.run(t_hist, dt, seed=SEED, stride=stride)

ts   = t_hist[::stride][:len(phi)]
x    = kz.grid()
idx  = [int(np.argmin(abs(ts - tt))) for tt in T_WIN]
snaps = [phi[i] for i in idx]
final = phi[-1]
pos   = kz.defect_positions(final)
n_def = kz.count_defects(final)

# ---- plot ----
fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

colors = ['tab:red', 'tab:blue', 'k']
widths = [2.0, 1.2, 0.8]
for s, i, c, w in zip(snaps, idx, colors, widths):
    ax[0].plot(x, s, color=c, lw=w,
               label=rf'$t={ts[i]:.0f}$  ($\epsilon={ts[i]/kz.TAU:.2f}$)')
ax[0].set_ylabel(r'$\phi$')
ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax[0].legend(fontsize=9)
ax[0].set_title(rf'input window, $\tau_Q={kz.TAU:.0f}$')

ax[1].plot(x, final, color='tab:green', lw=0.9,
           label=rf'$t={ts[-1]:.0f}$  ($\epsilon=1$)')
ax[1].plot(pos, np.zeros(len(pos)), 'o', mfc='none', mec='b', ms=6,
           label=f'{n_def} defects')
ax[1].axhline(0, color='grey', lw=0.5)
ax[1].set_xlabel('x'); ax[1].set_ylabel(r'$\phi$')
ax[1].set_xlim(0, kz.L())
ax[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('figure1.png', dpi=150)
plt.show()