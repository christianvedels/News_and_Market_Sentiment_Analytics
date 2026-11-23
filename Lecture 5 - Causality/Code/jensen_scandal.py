import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Nvidia around the Jensen Huang scandal, autumn 2026 (dates from TechCrunch's timeline)
events = {  # date: (label, label height)
    "2026-10-29": ("NYT report", 112),
    "2026-11-09": ("Admits affair", 112),
    "2026-11-11": ("Resigns", 106),
    "2026-11-17": ("Earnings", 112),
}
start, end = "2026-10-15", "2026-11-22"

df = yf.download(["NVDA", "^GSPC"], start=start, end=end, auto_adjust=True, progress=False)["Close"]
print((df.pct_change() * 100).round(1).loc["2026-10-27":].to_string())
df = df / df.iloc[0] * 100  # Normalize to 100 at start date

plt.figure(figsize=(6, 4))
plt.plot(df.index, df["NVDA"], label="Nvidia", color="C2", lw=2)
plt.plot(df.index, df["^GSPC"], label="S&P 500", color="C0")
plt.ylim(top=116)
for d, (label, y) in events.items():
    plt.axvline(pd.to_datetime(d), color="red", linestyle="--", linewidth=1)
    plt.text(pd.to_datetime(d), y, label, ha="center", va="bottom", fontsize=8,
             bbox=dict(facecolor="white", edgecolor="none", pad=1))
plt.legend(loc="lower left")
plt.title("Nvidia and the Jensen Huang scandal, autumn 2026")
plt.xlabel("Date")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Close price (15 Oct 2026 = 100)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Lecture 5 - Causality/Code/jensen_scandal_reaction.png")
