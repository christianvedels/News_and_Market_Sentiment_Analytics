---
editor_options: 
  markdown: 
    wrap: 72
---

# Week 2 Coding Challenge: The Zipf Mystery

**Submit here: [Challenge Submission Form](https://forms.gle/WmSEkZn8WH1fiDjE6)**
    
+ Each bullet solved: 5 points  
+ Submitting on time: 10 points  

**Additional requirement: Code must execute out of the box.**

## Challenge Details

### 1. In 35 minutes

- Define a function 'ZipfIt' which takes a textID ∈ (0, 1, ...) and
outputs the slope of the Zipf regression.

- Run this function on 'Texts': [Texts on
GitHub](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/tree/0610d97ed79eb0e2d92cb5f01290479f1bcf5d42/Code_challenge/Lecture%202/Texts)

- Use the outputs to classify real from fake texts. Test the quality
against 'Answers.csv'. You are not allowed to use anything in
'Answers.csv' as training data.

**Directories:** The code needs to run out of the box. In this case that means it needs to draw the texts from a local folder 'Texts', e.g. 'Texts/Text0.txt'. 

**Hint 1:** (One approach is to take [11_Zipf.py](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%202%20-%20Lexical%20resources/Code/11_Zipf.py) and turn it into
functions like 'load_text', 'clean_text', 'frequencies', etc. and then
tie it all together in 'ZipfIt'). This is the last few lines of such a
function:

**Hint 2:** Try plotting the histogram of the estimated slopes

**Hint 3:** It is possible to get 100% accuracy

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
