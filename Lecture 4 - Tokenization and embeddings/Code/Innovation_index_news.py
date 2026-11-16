"""
Innovation and novelty of financial news articles.

Reimplements the innovation index of Koschnick (2025), https://arxiv.org/abs/2512.16587
(package: https://github.com/juliuskoschnick/innovfromtext) in a few lines, with any embedding model.

Two measures for article i published in quarter t:

    innovation_i = f_k(i, next TAU quarters) / f_k(i, previous TAU quarters)
    novelty_i    = 1 - f_k(i, previous TAU quarters)          # past information only

f_k = mean of the k largest cosine similarities between article i and the articles in a pool.
Articles from quarter t itself are in neither pool.

Data: ~4,400 CNBC articles about S&P 500 firms, mostly 2017-2024 (MIT licence):
https://huggingface.co/datasets/KrossKinetic/SP500-Financial-News-Articles-Time-Series

Requirements: pip install torch transformers pandas matplotlib huggingface_hub
Runs in a few minutes on a laptop CPU. Next step: Innovation_predictive.py
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModel

# ---- Settings ----
START, END = "2017Q1", "2024Q1"     # quarters to use (2024Q2 is incomplete)
N_POOL = 60                         # articles per quarter in the reference pools. f_k rises with
                                    # pool size, so pools have equal size in every quarter
TAU = 4                             # window length in quarters (both directions)
K = 10                              # number of nearest neighbours averaged in f_k
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 20261116

# ---- 1. Load data ----
f = hf_hub_download("KrossKinetic/SP500-Financial-News-Articles-Time-Series",
                    "stock_data_articles.jsonl", repo_type="dataset")
raw = pd.read_json(f, lines=True)
raw["date"] = pd.to_datetime(raw["Publishdate"])
raw["q"] = raw["date"].dt.to_period("Q")
raw = raw[(raw["q"] >= START) & (raw["q"] <= END)]
raw = raw[raw["Text"].str.split().str.len() >= 50]        # drop stubs

# One row per article for the text measures (some articles are tagged to several firms)
df = raw.drop_duplicates("links").reset_index(drop=True)
print(f"{len(df):,} articles, {df['q'].nunique()} quarters")

# Equal-size reference pool per quarter
in_pool = (df.groupby("q").sample(frac=1, random_state=SEED)
             .groupby("q").head(N_POOL).index)
df["in_pool"] = df.index.isin(in_pool)

# ---- 2. Embed title + start of the article (mean pooling, L2-normalised) ----
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).eval()


@torch.no_grad()
def embed(texts, batch_size=32):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=256, return_tensors="pt")      # MiniLM reads up to 256 tokens
        h = model(**batch).last_hidden_state                  # (b, tokens, 384)
        mask = batch["attention_mask"].unsqueeze(-1)          # ignore padding
        v = (h * mask).sum(1) / mask.sum(1)                   # mean over tokens
        out.append(torch.nn.functional.normalize(v, dim=1))
        print(f"\rEmbedded {min(i + batch_size, len(texts)):,}/{len(texts):,}", end="")
    print()
    return torch.cat(out)


E = embed((df["Title"] + ". " + df["Text"]).tolist())  # unit vectors: dot product = cosine


# ---- 3. Measures ----
def f_k(Q, P, k=K):
    """Mean of the k largest cosine similarities between each row of Q and pool P."""
    return (Q @ P.T).topk(k, dim=1).values.mean(1)


q = df["q"].to_numpy()
pool = df["in_pool"].to_numpy()
quarters = sorted(df["q"].unique())
df["innovation"] = np.nan
df["novelty"] = np.nan
for j, t in enumerate(quarters):
    if j < TAU:
        continue                                          # not enough past
    i = np.where(q == t)[0]                               # score ALL articles of quarter t
    past = f_k(E[i], E[np.where(pool & (q >= t - TAU) & (q < t))[0]])
    df.loc[i, "novelty"] = (1 - past).numpy()
    if j + TAU < len(quarters):                           # complete future window
        fut = f_k(E[i], E[np.where(pool & (q > t) & (q <= t + TAU))[0]])
        df.loc[i, "innovation"] = (fut / past).numpy()

# ---- 4. Results ----
pd.set_option("display.max_colwidth", 95, "display.width", 200)
cols = ["q", "symbol", "innovation", "novelty", "Title"]
both = df.dropna(subset=["innovation", "novelty"])
print("\nMost INNOVATIVE (resemble the future more than the past):")
print(both.nlargest(10, "innovation")[cols].to_string(index=False))
print("\nMost NOVEL (least like the past):")
print(both.nlargest(10, "novelty")[cols].to_string(index=False))
print(f"\nCorrelation innovation vs novelty: {both['innovation'].corr(both['novelty']):.2f}")

# Figure: novelty vs innovation
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(both["novelty"], both["innovation"], s=6, alpha=0.3)
ax.axhline(1, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Novelty (1 - similarity to past 4 quarters)")
ax.set_ylabel("Innovation (future / past similarity)")
ax.set_title(f"CNBC articles on S&P 500 firms, {both['q'].min()}-{both['q'].max()}")
for _, r in pd.concat([both.nlargest(2, "innovation"), both.nlargest(2, "novelty")]).iterrows():
    ax.annotate(r["Title"][:40] + "...", (r["novelty"], r["innovation"]), fontsize=7,
                ha="right" if r["novelty"] > both["novelty"].median() else "left")
fig.tight_layout()
fig.savefig("Figures/innovation_vs_novelty.png", dpi=150)

# Save one row per article-firm pair (for the return analysis)
out = raw[["links", "symbol", "date", "q"]].merge(
    df[["links", "Title", "innovation", "novelty"]], on="links")
out.to_csv("Code/innovation_news_results.csv", index=False)

# Save embeddings + metadata (same row order) for Innovation_worked_examples.py
np.save("Code/article_embeddings.npy", E.numpy())
df[["links", "q", "in_pool", "Title", "innovation", "novelty"]].to_csv(
    "Code/article_meta.csv", index=False)
