"""Upward arrow for the Pearl's ladder slide: climbing requires more assumptions.
Run from the lecture folder."""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(1.6, 4.5))
ax.annotate("", xy=(0.5, 0.97), xytext=(0.5, 0.03),
            arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=1.2", lw=4, color="black"))
ax.text(0.3, 0.5, "More assumptions", rotation=90, ha="center", va="center", fontsize=14,
        fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.tight_layout()
fig.savefig("Figures/Assumptions_up.png", dpi=150, transparent=True)
