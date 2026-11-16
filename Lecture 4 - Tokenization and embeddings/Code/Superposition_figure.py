"""Illustration for the superposition slide: 5 sparse features stored in 2 dimensions
(the pentagon solution described in Elhage et al., 2022, Toy Models of Superposition,
https://transformer-circuits.pub/2022/toy_model/index.html). Run from the lecture folder."""

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))
angles = np.pi / 2 + 2 * np.pi * np.arange(5) / 5
for j, a in enumerate(angles):
    x, y = np.cos(a), np.sin(a)
    ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", lw=2, color=f"C{j}"))
    ax.text(1.2 * x, 1.2 * y, f"feature {j + 1}", ha="center", va="center", fontsize=11, color=f"C{j}")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.4, 1.5)
ax.set_aspect("equal")
ax.axhline(0, color="grey", lw=0.5)
ax.axvline(0, color="grey", lw=0.5)
ax.set_xlabel("dimension 1")
ax.set_ylabel("dimension 2")
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig("Figures/superposition_toy.png", dpi=150)
