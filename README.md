# News and Market Sentiment Analytics
This is the GitHub page of [DS821: News and Market Sentiment Analytics](https://odin.sdu.dk/sitecore/index.php?a=fagbesk&id=156413&lang=en) offered in the Data Science masters of the University of Southern Denmark. Here you can find slides and other content for the course. If you are a student of this course, please also check the itslearning page of the course.  

**Please note:** *The course is being developed throughout the semester. Please check back for updates.*
## Course book
The course is loosely based on:
Patel, A. A., & Arasanipalai, A. U. (2021). _Applied natural language processing in the enterprise_. O'Reilly Media. https://www.oreilly.com/library/view/applied-natural-language/9781492062561/

However, NLP is fast-moving these years. So although we will draw many examples from the textbook, we will also cover a few other things. 

## Lecture structure
Our lectures will generally follow this structure, which is roughly equally divided:  
**Part 1:** Theoretical introduction  
**Part 2:** Practical examples  
**Part 3:** Coding challenge (follow-up from last time + challenge based on today's lecture)

## Course plan and slides
| Date| Slides (links will appear)| Topic | Assigned Reading | Coding challenge |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ | 
| 2024-10-30 | [Lecture 1 - Introduction](https://raw.githack.com/christianvedels/News_and_Market_Sentiment_Analytics/refs/heads/main/Lecture%201%20-%20Introduction/Slides.html) | A broad overview of NLP + Setting up the course and expectations  | See below | [Coding_challenge 1](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%201%20-%20Introduction/Coding_challenge1.md) [Solution 1](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%201%20-%20Introduction/Coding_challenge_solution.py) |
| 2024-11-06 | [Lecture 2 - Classical data wrangling with text](https://raw.githack.com/christianvedels/News_and_Market_Sentiment_Analytics/refs/heads/main/Lecture%202%20-%20Data%20wrangling%20with%20text/Slides.html) | Classical data wrangling with text | [Text Wrangling with Python](https://blog.devgenius.io/text-wrangling-with-python-a-comprehensive-guide-to-nlp-and-nltk-f7553e713291) ['The Zipf Mystery'](https://youtu.be/fCn8zs912OE?si=xVMA63kt9M99Qvjx) | [Coding_challenge 2](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%202%20-%20Data%20wrangling%20with%20text/Coding_challenge_lecture2.md) [Solution 2](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%202%20-%20Data%20wrangling%20with%20text/Coding_challenge_solution.py) | 
| 2024-11-13 | [Lecture 3 - Classification pt 1](https://raw.githack.com/christianvedels/News_and_Market_Sentiment_Analytics/refs/heads/main/Lecture%203%20-%20Classification%20pt%201/Slides.html) | Classifcation - recipes for training a text classifier. I will also introduce an example from my own research, *OccCANINE*. | Ch 2, [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604) | [Coding_challenge 3](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%203%20-%20Classification%20pt%201/Coding_challenge_lecture3.md) [Solution 3](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%203%20-%20Classification%20pt%201/Coding_challenge_lecture3_solution.py) | 
| 2024-11-20 | [Lecture 4 - Classification pt 2](https://raw.githack.com//christianvedels/News_and_Market_Sentiment_Analytics/refs/heads/main/Lecture%204%20-%20Classification%20pt%202/Slides.html) | Classification - off-the shelf models | Ch 3 | [Coding_challenge 4](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%204%20-%20Classification%20pt%202/Coding_challenge_lecture4.md) [Solution 4 (by Tue Larsen)](https://youtu.be/gvysw5Hspx0?si=D3WLYb5Tdr20LPzB) |
| 2024-11-27 | [Lecture 5 - Tokenization and Embeddings](https://raw.githack.com//christianvedels/News_and_Market_Sentiment_Analytics/refs/heads/main/Lecture%205%20-%20Tokenization%20and%20embeddings/Slides.html) + [Guest lecture by Julius Koschnick](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%205%20-%20Tokenization%20and%20embeddings/Slides-Julius-Koschnick.pdf) | Tokenization and Embeddings | Ch 4, Ch 5, [3Blue1Brown: Attention in transformers, visually explained](https://youtu.be/eMlx5fFNoYc?si=HtfNuNlO4Y0e3olH) | [Coding_challenge 5](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%205%20-%20Tokenization%20and%20embeddings/Coding_challenge_lecture5.md); [Solution 5](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%205%20-%20Tokenization%20and%20embeddings/Coding_challenge_lecture5_solution.py); [Video: Solution 5](https://youtu.be/jxrIpeMis-g)
| 2024-12-04 | [Lecture 6 - Record Linking and Topic Models](https://raw.githack.com/christianvedels/News_and_Market_Sentiment_Analytics/refs/heads/main/Lecture%206%20-%20Record%20linking/Slides.html) | Record linking and topic modeling | Ch. 8; [Blogpost: Record Linkage with Machine Learning](https://blog.damavis.com/en/record-linkage-with-machine-learning/); [Automated Linking of Historical Data](https://www.nber.org/papers/w25825); [Topic modelling with LDA](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2) | [Coding challenge 6](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%206%20-%20Record%20linking/Coding_challenge_lecture6.md)
| 2024-12-11 | TBA | Sentiment analysis using RAGs | [Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models](https://arxiv.org/abs/2310.04027); [Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models](https://arxiv.org/abs/2306.12659)                                                                                                                                                                                                         |
| 2024-12-16 |  | Exam workshop. Will be held online (10:00-12:00). See itslearning for the zoom link. |                                                                                                                                                                                                              |


**Reading for lecture 1:**
For the first lecture you have three tasks to do beforehand:

1. Please read Chapter 1 of Patel & Arasanipalai (2021), which is freely available here: https://www.oreilly.com/library/view/applied-natural-language/9781492062561/ch01.html

2. Is ChatGPT high-tech plagiarism? https://youtu.be/SJi4VE-0MoA?si=sqFxbEGQ27drkvFA  

3. This course teaches NLP techniques applied to financial analysis. We can put more or less weight on the application. Please think about the type of skills you want to learn from this course. Then we will adapt the lectures accordingly.


## Contact information
**The course is taught by:**  
*Christian Vedel,*  
*christian-vs@sam.sdu.dk,*  
*Assistant Proffessor of Economics,*  
*Department of Economics*  
