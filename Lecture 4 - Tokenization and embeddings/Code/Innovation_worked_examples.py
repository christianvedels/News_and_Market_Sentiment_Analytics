"""
Worked examples: what goes into innovation and novelty for a single article?

Uses Code/article_embeddings.npy and Code/article_meta.csv from Innovation_index_news.py.
For readability we use k = 3 here (the main analysis uses k = 10).
"""

import numpy as np
import pandas as pd

K = 3
TAU = 4
EXAMPLES = [
    "Regeneron CEO hopes to have coronavirus treatment",   # high innovation
    "Queen Elizabeth's coffin reaches Windsor",            # high novelty
]

E = np.load("Code/article_embeddings.npy")
meta = pd.read_csv("Code/article_meta.csv")
meta["q"] = pd.PeriodIndex(meta["q"], freq="Q")
q = meta["q"].to_numpy()
pool = meta["in_pool"].to_numpy()

# Add a typical (low-novelty) article: the one with the lowest novelty
EXAMPLES.append(meta.loc[meta["novelty"].idxmin(), "Title"][:40])


def top_k(i, mask):
    """The k pool articles most similar to article i, with cosine similarities."""
    idx = np.where(mask)[0]
    sims = E[idx] @ E[i]                       # unit vectors: dot product = cosine
    best = idx[np.argsort(-sims)[:K]]
    return [(meta.at[j, "q"], E[j] @ E[i], meta.at[j, "Title"]) for j in best]


for text in EXAMPLES:
    i = meta.index[meta["Title"].str.startswith(text)][0]
    t = meta.at[i, "q"]
    print("=" * 100)
    print(f"[{t}] {meta.at[i, 'Title']}")
    past = top_k(i, pool & (q >= t - TAU) & (q < t))
    fut = top_k(i, pool & (q > t) & (q <= t + TAU))
    for name, nn in [("PAST", past), ("FUTURE", fut)]:
        print(f"  {name} {K} nearest:")
        for qq, s, title in nn:
            print(f"    {s:.2f}  [{qq}] {title[:85]}")
    fp = np.mean([s for _, s, _ in past])
    ff = np.mean([s for _, s, _ in fut])
    print(f"  f_k(past) = {fp:.2f}   f_k(future) = {ff:.2f}")
    print(f"  Novelty = 1 - {fp:.2f} = {1 - fp:.2f}   Innovation = {ff:.2f} / {fp:.2f} = {ff / fp:.2f}")
