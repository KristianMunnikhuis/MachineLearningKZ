import sys
sys.path.append("../")
import KZ_2D as kz
import matplotlib.pyplot as plt
import numpy as np

kz.TAU=128
t_hist, dt = kz.make_time_grid(n_steps=3200)
phi, pi = kz.run(t_hist, dt, n_real=None, seed=None, stride=1, phi0=None, pi0=None,
        check=True)

index_one = -900
index_two = -1
fig, ax = plt.subplots(1,2)
ax[0].imshow(phi[index_one], cmap='RdBu_r')
ax[0].contour(phi[index_one], levels=[0], colors='k', linewidths=1)

ax[1].imshow(phi[index_two], cmap='RdBu_r')
ax[1].contour(phi[index_two], levels=[0], colors='k', linewidths=1)
title_one=r"$\epsilon=$"+f"{t_hist[index_one]/kz.TAU}"[:4]
title_two=r"$\epsilon=$"+f"{t_hist[index_two]/kz.TAU}"[:4]

ax[0].set_title(title_one)
ax[1].set_title(title_two)
fig.suptitle(r"$\tau=$"+f"{kz.TAU}")

plt.savefig("out/2d_contour_map.pdf",dpi=800)