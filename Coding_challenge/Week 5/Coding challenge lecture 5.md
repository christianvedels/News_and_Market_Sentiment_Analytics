# Week 5 Coding Challenge: Predicting volatility

This weeks coding challenge is of a very open-ended nature. You are asked to extract information on risk from news sources. 

One thing is to predict stock prices, another issue is to estimate uncertainty. In very uncertain markets it is advisable to divest into safer assets with lower expected return. One simple measure of volatility is the standard deviation of the S&P500 on a window of 30 days. Please read more about volalitity prediction before doing something like this with any real money. Oversimplified statistical measures like this is in part to blame for the millions of people adversely affected by the 2008 Financial crisis. With that cautionary tale out of the way, it works well for learning something about applying NLP tools to gain financial insights.

**Submit here: [Challenge Submission Form](https://forms.gle/WmSEkZn8WH1fiDjE6)**

## Challenge Details

### 1. In 35 minutes (and until next lecture)

- Register for a free news api key at https://newsapi.org/
- Extract relevant features from relevant news for market uncertainty 
- Get the S&P500 index at a **daily** frequency
- Estimate volatility with a 30-day window 
- **Use your data to build a regression model that predicts volatility**
- The output of the regression model is a news-volatility index. How does it correlate with volalitity?

## Resources
- Check [Lecture 2 - Lexical resources/Code/01_Get_stock_data.py](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%202%20-%20Lexical%20resources/Code/01_Get_stock_data.py)  
- Get news and extract features with [Lecture 5 - Understanding and utilizing grammar/Code/Getting_recent_news.py](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%205%20-%20Understanding%20and%20utilizing%20grammar/Code/Getting_recent_news.py)  
- Use the following formula to estimate volatility. Please note the cautionary note above:

$$\hat{\sigma}_{t,t-30} = \sqrt{\sum_{i=t-30}^{t}\left[\ln(p_i/p_{i-1})\right]^2}$$