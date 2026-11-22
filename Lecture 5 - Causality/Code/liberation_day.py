import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Market indices to track broader market reaction
# ^DJI = Dow Jones Industrial Average
# ^GSPC = S&P 500
# ^IXIC = NASDAQ Composite
tickers = ["^DJI", "^GSPC", "^IXIC"]

# Trump's "Liberation Day" tariff announcement
# Event window around the announcement (April 2, 2025)
start = "2025-01-01"
end = "2025-06-30"
# Use yfinance Tickers class to handle multiple tickers
tickers_yf = yf.Tickers(" ".join(tickers))
# Retrieve history for each ticker
history = tickers_yf.history(interval="1d", start=start, end=end)

# Extract closing prices
df = history['Close']

# Normalize
df = df / df.iloc[0] * 100  # Normalize to 100 at start date

# Plot with vertical line on Liberation Day
plt.figure(figsize=(6, 4))
for t in tickers:
    label_map = {"^DJI": "Dow Jones", "^GSPC": "S&P 500", "^IXIC": "NASDAQ"}
    plt.plot(df.index, df[t], label=label_map.get(t, t))

plt.legend()
plt.title("Trump's 'Liberation Day' Tariff Announcement")
plt.xlabel("Date")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Close Price")
plt.axvline(pd.to_datetime("2025-04-02"), color='red', linestyle='--', linewidth=2, label="Liberation Day")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Lecture 5 - Causality/Code/liberation_day_market_reaction.png")
plt.show()