"""
Do innovation and novelty of a news article predict the firm's stock return afterwards?

Input: Code/innovation_news_results.csv from Innovation_index_news.py
Prices: Yahoo Finance via yfinance (pip install yfinance)

For each article about firm s published on day d (date only, no time of day):
    abnormal return = log return of s minus log return of SPY (S&P 500 ETF)
    over a window counted in trading days from day 0 = the last trading day on or before d:

    next_day:   close of day 0 -> close of day 1
                (catches the reaction to articles published after the close;
                 for articles published during trading hours, part of the reaction is missed)
    next_month: close of day 1 -> close of day 22
                (skips the reaction days entirely)

Performance statistic: information coefficient (IC) = Spearman rank correlation, within each
quarter, between the measure and the abnormal return (and its absolute value).
Mean IC over quarters with t-statistic = mean / sd * sqrt(#quarters).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

WINDOWS = {  # name: (first day, last day), in trading days after day 0
    "next_day": (0, 1),
    "next_month": (1, 22),
}
LABELS = {"next_day": "Next-day", "next_month": "Next-month"}

articles = pd.read_csv("Code/innovation_news_results.csv", parse_dates=["date"])
articles["symbol"] = articles["symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B

# ---- 1. Prices ----
tickers = sorted(articles["symbol"].unique()) + ["SPY"]
px = yf.download(tickers, start="2016-12-01", end="2024-07-01",
                 auto_adjust=True, progress=False)["Close"]
logpx = np.log(px)
days = logpx.index
day0 = days.searchsorted(articles["date"], side="right") - 1   # last trading day <= d


# ---- 2. Abnormal returns ----
def abnormal_returns(a, b):
    ok = (day0 + b < len(days)) & (day0 >= 0)
    d = articles[ok].copy()
    s0, s1 = days[day0[ok] + a], days[day0[ok] + b]
    r = np.array([logpx.at[t1, s] - logpx.at[t0, s] if s in logpx else np.nan
                  for s, t0, t1 in zip(d["symbol"], s0, s1)])
    spy = logpx["SPY"].reindex(s1).to_numpy() - logpx["SPY"].reindex(s0).to_numpy()
    d["ar"] = r - spy
    d["abs_ar"] = d["ar"].abs()
    return d.dropna(subset=["ar"])


# ---- 3. Quarterly information coefficients ----
def ic_series(df, measure, target):
    d = df.dropna(subset=[measure])
    return d.groupby("q").apply(lambda g: g[measure].corr(g[target], method="spearman"))


def summary(ic):
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)), len(ic)


PERIODS = {  # name: (first year, last year)
    "all": (2018, 2024),
    "2018_2019": (2018, 2019),   # before Covid
    "2020_2021": (2020, 2021),   # Covid
    "2022_2024": (2022, 2024),   # rate hikes, AI boom
}


def scatter_figure(df, title, path):
    """2 x 2 scatter: (innovation, novelty) x (AR, |AR|), with decile means and Spearman."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for row, measure in enumerate(["innovation", "novelty"]):
        d = df.dropna(subset=[measure])
        for col, target in enumerate(["ar", "abs_ar"]):
            ax = axes[row, col]
            y = 100 * d[target]                                  # in percent
            ax.scatter(d[measure], y, s=4, alpha=0.25, color=f"C{row}")
            dec = pd.qcut(d[measure], 10)
            means = y.groupby(dec, observed=True).mean()
            centers = d[measure].groupby(dec, observed=True).mean()
            ax.plot(centers, means, color="black", marker="o", ms=4, label="Decile means")
            ax.set_ylim(np.percentile(y, [1, 99]))               # hide extreme outliers
            ax.axhline(0, color="grey", lw=0.8)
            ax.set_xlabel(measure.capitalize())
            rho = d[measure].corr(d[target], method="spearman")
            ax.set_title(f"{'|AR|' if target == 'abs_ar' else 'AR'} (%), "
                         f"Spearman = {rho:+.3f} (n = {len(d):,})", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            print(f"  {title[-9:]} {measure:>10} -> {target:<6}: Spearman = {rho:+.3f}, n = {len(d)}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


for name, (a, b) in WINDOWS.items():
    df = abnormal_returns(a, b)
    print(f"\n== {name}: {len(df):,} article-firm pairs, {df['symbol'].nunique()} firms")
    results = {}
    for target in ["ar", "abs_ar"]:
        for measure in ["innovation", "novelty"]:
            ic = ic_series(df, measure, target)
            results[(measure, target)] = ic
            m, t, n = summary(ic)
            print(f"{measure:>10} -> {target:<6}: mean IC = {m:+.3f}, t = {t:+.2f}, quarters = {n}")

    # ---- 4. Figure: IC per quarter for both measures ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, target in zip(axes, ["ar", "abs_ar"]):
        for measure, c in [("innovation", "C0"), ("novelty", "C1")]:
            ic = results[(measure, target)]
            m, t, _ = summary(ic)
            ax.plot(ic.index.astype(str), ic.values, marker="o", ms=3, color=c,
                    label=f"{measure}: mean IC {m:+.3f} (t = {t:.1f})")
            ax.axhline(m, color=c, ls=":", lw=1)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{LABELS[name]} {'|abnormal return|' if target == 'abs_ar' else 'abnormal return'}")
        ax.set_xticks(ax.get_xticks()[::4])
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Spearman IC within quarter")
    fig.tight_layout()
    fig.savefig(f"Figures/innovation_predictive_ic_{name}.png", dpi=150)
    plt.close(fig)

    # ---- 5. Figures: scatter of measure vs return, with decile means, all years and by period ----
    for period, (p0, p1) in PERIODS.items():
        sub = df[(df["date"].dt.year >= p0) & (df["date"].dt.year <= p1)]
        scatter_figure(sub, f"{LABELS[name]} abnormal return (AR) vs innovation and novelty, {p0}-{p1}",
                       f"Figures/innovation_scatter_{name}_{period}.png")
