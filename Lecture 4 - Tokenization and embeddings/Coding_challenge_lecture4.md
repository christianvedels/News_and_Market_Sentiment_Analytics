# Week 4 Embeddings and zero shot classifiers

In these weeks coding challenge you are asked to use embeddings to build a zero-shot classifier

1. Find a sensible model to get word and sentence embeddings on HuggingFace
2. Would you expect the mean of the embeddings of words in a sentence to be approximately equal the embedding of the entire sentence? Why? Please test with these sentences:
   + "He banked the plane sharply to avoid the ridge."
   + "He was a master banker because he built the best banks in the trench."
   + "He was a master banker - all accounts were kept in order."
   + "After hearing the sentence, he felt a mix of relief and regret, knowing that justice had been served."
   + "This is just a regular old sentence."  
   (Please derive and use proper metric to test the similarity - work off the hint below)
2. What does your results for 2 tell you about the effectiveness of bag of word models?
3. Build a zero shot classifier using cosine distance. And use it to to repeat [coding challenge 3](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%203%20-%20Classification%20pt%201/Coding_challenge_lecture3.md)

References regarding zero-shot classification: 
- https://arxiv.org/abs/1909.00161
- https://joeddav.github.io/blog/2020/05/29/ZSL.html 


### Hint 1 (for sentence versus word embeddings)

For a sentence with tokens $w_1, \dots, w_n$ and token embeddings $e_i = f(w_i)$, the mean word embedding is

$$\bar e = \frac{1}{n} \sum_{i=1}^n e_i.$$

Many models instead output a sentence embedding $s = f(\text{sentence})$ via a special token or pooling. You can compare how close these are with cosine similarity

$$\cos(\bar e, s) = \frac{\bar e \cdot s}{\|\bar e\| \,\|s\|}.$$

*If* $\bar{e} = s$ *then* $\cos(\bar e, s) = 1$. Is this realistic for the sentences above?


### Hint 2 (for the zero-shot classifier)
Given an input sentence $x$ with embedding $v_x = f(x)$ and a set of label prompts $\ell_1, \dots, \ell_K$ (e.g. “This news is positive.”, “This news is negative.”), compute label embeddings

$$
v_k = f(\ell_k), \quad k = 1, \dots, K.
$$

The cosine similarity between $x$ and label $k$ is

$$
\cos(v_x, v_k) = \frac{v_x \cdot v_k}{\|v_x\| \,\|v_k\|}.
$$

A simple zero-shot classifier chooses the label with highest cosine similarity:

$$
\hat{k}(x) = \arg\max_{k \in \{1, \dots, K\}} \cos(v_x, v_k).
$$


