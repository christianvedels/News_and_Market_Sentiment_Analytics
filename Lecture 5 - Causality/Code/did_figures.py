"""
Difference-in-differences illustration with the numbers from Card & Krueger (1994), Table 3:
average full-time-equivalent (FTE) employment per fast-food restaurant.
https://www.nber.org/papers/w4509
Run from the lecture folder.
"""

import matplotlib.pyplot as plt

nj = {"before": 20.44, "after": 21.03}   # New Jersey (minimum wage raised April 1992)
pa = {"before": 23.33, "after": 21.17}   # Pennsylvania (no change)

counterfactual = nj["before"] + (pa["after"] - pa["before"])
did = nj["after"] - counterfactual
print(f"Change NJ: {nj['after'] - nj['before']:+.2f}, change PA: {pa['after'] - pa['before']:+.2f}")
print(f"DiD = {did:+.2f} FTE per store")

x = [0, 1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(x, [pa["before"], pa["after"]], "o-", color="C0", lw=2.5, label="Pennsylvania (control)")
ax.plot(x, [nj["before"], nj["after"]], "o-", color="C3", lw=2.5, label="New Jersey (treated)")
ax.plot(x, [nj["before"], counterfactual], "o--", color="C3", alpha=0.5, lw=2,
        label="New Jersey without the increase\n(counterfactual: parallel trend)")
ax.annotate("", xy=(1.03, nj["after"]), xytext=(1.03, counterfactual),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax.text(1.06, (nj["after"] + counterfactual) / 2, f"DiD = {did:+.2f}", va="center", fontsize=11)
ax.set_xticks(x, ["Feb 1992\n(before)", "Nov 1992\n(after)"])
ax.set_xlim(-0.2, 1.35)
ax.set_ylabel("FTE employment per store")
ax.set_title("Card & Krueger (1994): minimum wage and fast-food employment")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig("Figures/did_card_krueger.png", dpi=150)

# ---- Parallel trends: holds vs violated (made-up data) ----
import numpy as np

t = np.arange(-4, 5)                    # periods; treatment starts at t = 0
control = 10 + 0.5 * t
effect = np.where(t >= 0, 2.0, 0.0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# (a) Parallel trends hold: same slope before treatment
treated = 13 + 0.5 * t + effect
cf = 13 + 0.5 * t
ax = axes[0]
ax.plot(t, control, "o-", color="C0", label="Control")
ax.plot(t, treated, "o-", color="C3", label="Treated")
ax.plot(t[t >= -1], cf[t >= -1], "--", color="C3", alpha=0.5, label="Treated without treatment\n(counterfactual)")
ax.annotate("", xy=(4.15, treated[-1]), xytext=(4.15, cf[-1]),
            arrowprops=dict(arrowstyle="<->", lw=1.5))
ax.text(4.3, (treated[-1] + cf[-1]) / 2, "Effect", va="center")
ax.set_title("Parallel trends hold: DiD finds the effect")

# (b) Violated: the treated group was already growing faster
slope = 0.9
treated_v = 13 + slope * t + effect
cf_true = 13 + slope * t
cf_did = treated_v[t == -1][0] + 0.5 * (t - (-1))    # what DiD assumes
ax = axes[1]
ax.plot(t, control, "o-", color="C0", label="Control")
ax.plot(t, treated_v, "o-", color="C3", label="Treated")
ax.plot(t[t >= -1], cf_true[t >= -1], "--", color="C3", alpha=0.5, label="True counterfactual")
ax.plot(t[t >= -1], cf_did[t >= -1], ":", color="black", label="Counterfactual DiD assumes")
ax.set_title("Violated: treated was already pulling away\nDiD overstates the effect")

for ax in axes:
    ax.axvline(-0.5, color="grey", ls=":")
    ax.set_xlabel("Period (treatment starts at 0)")
    ax.set_xlim(-4.5, 5.2)
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("Outcome")
fig.tight_layout()
fig.savefig("Figures/parallel_trends.png", dpi=150)
