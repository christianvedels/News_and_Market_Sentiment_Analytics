# Week 2 Coding Challenge: The Zipf Mystery

## Challenge Details

### 1. Zipf regressions

- Define a function 'ZipfIt' which takes a textID ∈ (0, 1, ...) and
outputs the slope of the Zipf regression. 

- Run this function on 'Texts': [Texts on
GitHub](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/tree/main/Lecture%202%20-%20Data%20wrangling%20with%20text/Coding_challenge_data/Texts)

- Use the outputs to classify real from fake texts. Test the quality
against ['Answers.csv'](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%202%20-%20Data%20wrangling%20with%20text/Coding_challenge_data/Answer.csv). You are not allowed to use anything in
'Answers.csv' as training data.

**Hint 1:** ChatGPT is handy: https://chatgpt.com/share/672a8a48-f9d8-800c-87fc-fc09c1e3c9e1 

**Hint 2:** Try plotting the histogram of the estimated slopes

## Notes on Zipf's Law

**Zipf's law** states that in natural languages, there is a specific
pattern of word frequencies. Namely:

$$F_n = \frac{F_1}{n}$$

That is, if we order all words by frequency, the frequency of the $n^{th}$
word, $F_n$, is equal to the frequency of the most common word divided by
the rank of that word.

Something which is useful is that the above equation implies that the
following linear relationship which fits into a linear regression:

$$\log(F_n) = \log(F_1) - \log(n)$$

$$\log(F_n) = \beta_0 - \beta_1 \log(n) + \varepsilon_n$$

Where $\beta_1 = 1$ and $\mathbb{E}(\varepsilon_n) = 0$

(In practice $\beta_1$ is often slightly different from 1)
