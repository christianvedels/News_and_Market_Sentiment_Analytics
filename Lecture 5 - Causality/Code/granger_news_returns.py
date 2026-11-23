"""
Granger causality: does daily news sentiment predict S&P 500 returns, or the other way around?

Data:
  - ~1.8M stock headlines 2009-2020 (Seeking Alpha, Zacks, ...):
    https://huggingface.co/datasets/ashraq/financial-news
  - S&P 500 (SPY) prices from Yahoo Finance
Sentiment: AFINN lexicon (Lecture 2), averaged over all headlines of a day.

Requirements: pip install pandas pyarrow afinn statsmodels yfinance matplotlib huggingface_hub
Run from the lecture folder.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from afinn import Afinn
from huggingface_hub import hf_hub_download
from statsmodels.tsa.stattools import grangercausalitytests

START, END = "2010-01-01", "2020-05-31"
MAXLAG = 5

# ---- 1. Daily news sentiment ----
f = hf_hub_download("ashraq/financial-news",
                    "data/train-00000-of-00001-8ec327f23bbe0948.parquet", repo_type="dataset")
news = pd.read_parquet(f, columns=["headline", "date"])
news["date"] = pd.to_datetime(news["date"], format="mixed").dt.normalize()
news = news[(news["date"] >= START) & (news["date"] <= END)].drop_duplicates("headline")
afinn = Afinn()
news["sent"] = news["headline"].map(afinn.score)
print(f"{len(news):,} headlines")

# ---- 2. Returns ----
spy = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)["Close"]["SPY"]
ret = np.log(spy).diff().dropna()
days = ret.index

# Headlines from weekends/holidays count towards the next trading day
news["day"] = days[np.minimum(days.searchsorted(news["date"]), len(days) - 1)]
daily = news.groupby("day")["sent"].mean()

df = pd.DataFrame({"R": 100 * ret, "S": daily}).dropna()
df["S"] = (df["S"] - df["S"].mean()) / df["S"].std()      # standardised sentiment
print(f"{len(df):,} trading days, {df.index.min().date()} to {df.index.max().date()}")
print(f"Same-day correlation of sentiment and return: {df['S'].corr(df['R']):.2f}")


# ---- 3. Granger tests in both directions ----
def granger_p(data, maxlag=MAXLAG):
    """p-values of the F-test that the 2nd column Granger-causes the 1st, lags 1..maxlag."""
    res = grangercausalitytests(data, maxlag=maxlag)
    return pd.Series({lag: res[lag][0]["ssr_ftest"][1] for lag in res})


out = pd.DataFrame({
    "Sentiment -> returns": granger_p(df[["R", "S"]]),
    "Returns -> sentiment": granger_p(df[["S", "R"]]),
})
out.index.name = "lags"
print("\nGranger F-test p-values (H0: no predictive power)")
print(out.map(lambda p: f"{p:.3g}").to_string())


# Size of the effect: regress S_t on own lags + lag of returns, and vice versa (1 lag)
def lag_coef(y, x):
    d = pd.DataFrame({"y": df[y], "y1": df[y].shift(), "x1": df[x].shift()}).dropna()
    X = np.column_stack([np.ones(len(d)), d["y1"], d["x1"]])
    b, *_ = np.linalg.lstsq(X, d["y"], rcond=None)
    return b[2]


print(f"\nYesterday's return +1 pp -> today's sentiment {lag_coef('S', 'R'):+.3f} sd")
print(f"Yesterday's sentiment +1 sd -> today's return {lag_coef('R', 'S'):+.3f} pp")

# ---- 4. Figure: 2020 crash, sentiment and returns ----
w = df.loc["2019-12-01":"2020-05-31"]
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.bar(w.index, w["R"], color="grey", alpha=0.6, label="S&P 500 return (%)")
ax1.set_ylabel("S&P 500 daily return (%)")
ax2 = ax1.twinx()
ax2.plot(w.index, w["S"].rolling(5).mean(), color="C3", lw=2, label="News sentiment (5-day mean, sd)")
ax2.set_ylabel("News sentiment (sd)", color="C3")
ax1.set_title("Daily returns and headline sentiment around the Covid crash")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig("Figures/granger_covid.png", dpi=150)
