import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Quantum-computing stocks around Nvidia CEO Jensen Huang's remark at CES
# (evening of 7 January 2025) that "very useful" quantum computers are 15-30 years away
tickers = ["IONQ", "RGTI", "QBTS", "QUBT"]
labels = {"IONQ": "IonQ", "RGTI": "Rigetti", "QBTS": "D-Wave", "QUBT": "Quantum Computing Inc."}
start = "2024-12-01"
end = "2025-02-28"

df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
df = df / df.iloc[0] * 100  # Normalize to 100 at start date

plt.figure(figsize=(6, 4))
for t in tickers:
    plt.plot(df.index, df[t], label=labels[t])

plt.legend()
plt.title("Quantum stocks and Jensen Huang's remark (CES, 7 Jan 2025)")
plt.xlabel("Date")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Close price (1 Dec 2024 = 100)")
plt.axvline(pd.to_datetime("2025-01-07"), color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Lecture 5 - Causality/Code/jensen_quantum_reaction.png")
