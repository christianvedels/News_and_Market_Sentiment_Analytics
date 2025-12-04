
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Magnificent 7 tickers (one common definition)
# Major AI-related tech companies
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

# Sam Altman "AI bubble" quote date
# Event window around the statement
start = "2025-07-01"
# Use yfinance Tickers class to handle multiple tickers
tickers_yf = yf.Tickers(" ".join(tickers))
# Retrieve history for each ticker
history = tickers_yf.history(period = "3mo", interval = "1d", start=start)

# Extract closing prices
df = history['Close']

# Normalize to 100 at start date
df = df / df.iloc[0] * 100

# Plot with vertical line on the quote date
plt.figure(figsize=(6, 4))
for t in tickers:
    plt.plot(df.index, df[t], label=t)

plt.legend()
plt.title("Sam Altman's AI Bubble Quote")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.axvline(pd.to_datetime("2025-08-15"), color='red', linestyle='--', linewidth=2, label="AI Bubble Quote Date")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Lecture 5 - Causality/Code/ai_bubble_quote_reaction.png")
plt.show()