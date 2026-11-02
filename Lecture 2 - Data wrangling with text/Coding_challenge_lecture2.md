# Coding challenge 2: The Zipf Mystery

## Challenge Details

This challenge is all about practicing some text wrangling skills. We will apply it to Zipf's law and show that using knowledge of the structure of language can be useful. 

### Zipf regressions
*In this exercise you will use the Zipf-property of natural language to detect synthetically created text*

- Define a function 'ZipfIt' which takes a textID ∈ {0, 1, ..., 999} and
outputs the slope of the Zipf regression. 

- Run this function on 'Texts': [Texts on
GitHub](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/tree/main/Lecture%202%20-%20Data%20wrangling%20with%20text/Coding_challenge_data/Texts)

- Use the outputs to classify real from fake texts. Test the quality
against ['Answer.csv'](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%202%20-%20Data%20wrangling%20with%20text/Coding_challenge_data/Answer.csv). You are not allowed to use anything in
'Answer.csv' as training data.



**Hint 1:** A chatbot or coding agent is handy here, and free tiers are enough. Ask it to explain each step, make sure you can run and explain the result, and say how you used it if you present.

**Hint 2:** Try plotting the histogram of the estimated slopes

## Notes on Zipf's Law

**Zipf's law** states that in natural languages, there is a specific
pattern of word frequencies. Namely:

$$F_n = \frac{F_1}{n}$$

That is, if we order all words by frequency, the frequency of the $n^{th}$
word, $F_n$, is equal to the frequency of the most common word divided by
the rank of that word.

Something which is useful is that the above equation implies the
following linear relationship, which we can estimate with a linear regression:

$$\log(F_n) = \log(F_1) - \log(n)$$

$$\log(F_n) = \beta_0 - \beta_1 \log(n) + \varepsilon_n$$

Where $\beta_1 = 1$ and $\mathbb{E}(\varepsilon_n) = 0$

(In practice $\beta_1$ is often slightly different from 1)
