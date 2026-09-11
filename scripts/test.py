import numpy as np
import matplotlib.pyplot as plt

d = np.load("data/2D/chunk_000.npz", allow_pickle=True)
snaps = d["snaps"]
for key in d.files:
    print(key, d[key].shape)

print("snap_times:", d["snap_times"])

rng = np.random.default_rng()
picks = rng.choice(len(snaps), size=4, replace=False)   # 4 different samples

fig, axes = plt.subplots(4, 4, figsize=(14, 14))

for row, n in enumerate(picks):
    for col, i in enumerate([5, 16, 27]):
        axes[row, col].imshow(snaps[n, i], cmap="RdBu")
        axes[row, col].set_title(f"sample {n}, t = {d['snap_times'][i]}")

    axes[row, 3].imshow(d["phi_final"][n], cmap="RdBu")
    axes[row, 3].set_title(f"sample {n}, final")

plt.savefig("test.png", dpi=200)

print("samples:", picks)
print("wall lengths:", d["wall_len"][picks])
