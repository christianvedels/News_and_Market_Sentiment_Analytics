# Coding challenge 5: Did ChatGPT move AI-exposed stocks?

In the lecture, difference-in-differences (DiD) measured how AI affected workers. Now you use it on the stock market, with a treatment variable that you build from text.

**The question:** When ChatGPT launched (30 November 2022), did the stocks of S&P 500 firms whose business is close to generative AI move differently from the rest? What about when DeepSeek R1 shocked the market (27 January 2025)?

Everything runs on free tools on a laptop (about 5–10 minutes, mostly downloading).

```
pip install pandas lxml requests yfinance torch transformers statsmodels matplotlib
```

## Tasks

### 1. Get the firms and their business descriptions

The snippet downloads the list of S&P 500 companies from Wikipedia and each firm's business description from Yahoo Finance. It caches the result, so you only download once.

```python
import io, os
import numpy as np
import pandas as pd
import requests
import yfinance as yf

if os.path.exists("descriptions.csv"):
    firms = pd.read_csv("descriptions.csv")
else:
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0 (course material)"}).text
    firms = pd.read_html(io.StringIO(html))[0][["Symbol", "Security", "GICS Sector"]]
    firms["Symbol"] = firms["Symbol"].str.replace(".", "-", regex=False)   # BRK.B -> BRK-B
    firms["description"] = [yf.Ticker(t).info.get("longBusinessSummary") for t in firms["Symbol"]]
    firms.to_csv("descriptions.csv", index=False)

firms = firms.dropna(subset=["description"]).reset_index(drop=True)
```

### 2. Measure AI exposure from text

Write a short description of what generative AI is and does. Embed it and every business description with a sentence-embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`; see the code from Lecture 4). Exposure = cosine similarity, stored in `firms["exposure"]`. Which firms come out on top? Does that make sense?

### 3. Define treated and control

Treated = the top third by exposure; control = the bottom third.

```python
firms["group"] = pd.qcut(firms["exposure"], 3, labels=["low", "mid", "high"])
firm_group = firms.set_index("Symbol")["group"]
```

### 4. Abnormal returns

Abnormal return = the firm's daily log return minus that of the S&P 500 (`SPY`), in percentage points.

```python
px = yf.download(firms["Symbol"].tolist() + ["SPY"], start="2022-06-01", end="2025-04-30",
                 auto_adjust=True)["Close"]
ret = np.log(px).diff()
ar = ret.drop(columns="SPY").sub(ret["SPY"], axis=0) * 100
```

### 5. Build the event-study panel

Take a window around the event: 40 trading days before to 120 after ChatGPT, 20 before to 20 after DeepSeek. For each firm, the outcome is the cumulative abnormal return since the start of the window (the firm's log price relative to the market), kept at the end of each week. Weeks are numbered relative to the event, so week −1 is the week before.

```python
def make_panel(date, before, after):
    day0 = ar.index.searchsorted(pd.Timestamp(date))
    win = ar.iloc[day0 - before: day0 + after + 1]
    win = win.loc[:, win.notna().mean() > 0.95].fillna(0)            # firms traded throughout
    week = np.floor((np.arange(len(win)) - before) / 5).astype(int)  # days -5..-1 = week -1
    weekly = win.cumsum().groupby(week).last()
    weekly.index.name = "week"
    panel = weekly.reset_index().melt(id_vars="week", var_name="firm", value_name="car")
    panel["group"] = panel["firm"].map(firm_group)
    panel = panel[panel["group"].isin(["high", "low"])].copy()
    panel["treated"] = (panel["group"] == "high").astype(int)
    return panel

panel = make_panel("2022-11-30", before=40, after=120)    # ChatGPT
```

### 6. Event-study regression

Run one regression with a separate $\beta_k$ for every week, just as the papers in the lecture do:

$$CAR_{ik} = \alpha_i + \gamma_k + \sum_{k \neq -1} \beta_k \cdot \text{Treated}_i \times 1[\text{week} = k] + \varepsilon_{ik}$$

It has firm fixed effects ($\alpha_i$) and week fixed effects ($\gamma_k$), and standard errors clustered by firm. Week −1 is the reference, so $\beta_{-1} = 0$.

```python
import statsmodels.formula.api as smf

weeks = sorted(panel["week"].unique())
terms = []
for k in weeks:
    if k != -1:                                                  # week -1 is the reference
        name = f"d{k}".replace("-", "m")                         # d3, dm2, ...
        panel[name] = ((panel["week"] == k) & (panel["treated"] == 1)).astype(int)
        terms.append(name)

fit = smf.ols("car ~ " + " + ".join(terms) + " + C(firm) + C(week)", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["firm"].astype("category").cat.codes})

es = pd.DataFrame({"week": [k for k in weeks if k != -1],
                   "beta": fit.params[terms].values, "se": fit.bse[terms].values})
```

### 7. Plot and read it

Plot $\beta_k$ against $k$ with 95% confidence intervals ($\pm 1.96 \cdot$ se). Remember to add the reference week, $\beta_{-1} = 0$.

- Are the $\beta_k$ before the event close to zero (parallel pre-trends)?
- What happens after?

Then repeat steps 5–7 for DeepSeek: `make_panel("2025-01-27", before=20, after=20)`.

## Discuss

- Do the pre-trends look parallel for each event? Which result do you trust more, and why?
- Is the effect a jump on the day, or a slow drift? What does that tell you about how fast markets understood each event?
- Your exposure measure picks up AI *producers* (chips, software) as well as firms whose work AI could *replace*. Why does that matter for interpreting $\beta_k$?
- You used **today's** S&P 500 list and **today's** business descriptions. Name two biases this introduces. (Hint: survivorship, and the look-ahead bias from Lecture 4.)
- What else happened around each date that could explain your results?
- Bonus: Try a different AI description, or use the top and bottom 10% instead of thirds. How robust is the picture?
