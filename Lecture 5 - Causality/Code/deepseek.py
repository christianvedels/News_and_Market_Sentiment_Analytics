import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# AI-exposed companies most affected by DeepSeek announcement
# NVDA was hit hardest with record $590B one-day loss
# Other major AI/chip/cloud infrastructure players
tickers = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL"]

# DeepSeek AI "Sputnik moment" - 27 January 2025
# Chinese AI startup launched ultra-cheap, high-efficiency model
# Triggered largest one-day company loss in Wall Street history (NVDA ~$590B)
# Event window around the announcement
start = "2025-01-01"
# Use yfinance Tickers class to handle multiple tickers
tickers_yf = yf.Tickers(" ".join(tickers))
# Retrieve history for each ticker
history = tickers_yf.history(period = "2mo", interval = "1d", start=start)

# Extract closing prices
df = history['Close']

# Normalize to 100 at start date
df = df / df.iloc[0] * 100

# Plot with vertical line on DeepSeek announcement date
plt.figure(figsize=(6, 4))
for t in tickers:
    linewidth = 3 if t == "NVDA" else 2  # Highlight NVDA (biggest loser)
    plt.plot(df.index, df[t], label=t, linewidth=linewidth)

plt.legend()
plt.title("AI/Tech Stock Prices Around DeepSeek 'Sputnik Moment'\n(NVDA lost record $590B in one day)")
plt.xlabel("Date")
plt.ylabel("Normalized Price (Jan 1, 2025 = 100)")
plt.axvline(pd.to_datetime("2025-01-27"), color='red', linestyle='--', linewidth=2, label="DeepSeek Announcement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Lecture 5 - Causality/Code/deepseek_ai_shock_reaction.png")
plt.show()