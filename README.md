# News and Market Sentiment Analytics
This is the GitHub page of [DS821: News and Market Sentiment Analytics](https://odin.sdu.dk/sitecore/index.php?a=fagbesk&id=156413&lang=en) offered in the Data Science masters of the University of Southern Denmark. Here you can find slides and other content for the course. If you are a student of this course, please also check the itslearning page of the course.  

The course runs in parallel in **Kolding** and **Odense** in the autumn of 2026, with the same content on both campuses.

**Please note:** *The course is being developed throughout the semester, and the plan below will be revised along the way. Please check back for updates.*

## Course book
The course is loosely based on:
Patel, A. A., & Arasanipalai, A. U. (2021). _Applied natural language processing in the enterprise_. O'Reilly Media. https://www.oreilly.com/library/view/applied-natural-language/9781492062561/

However, NLP is fast-moving these years. So although we will draw many examples from the textbook, we will not at all follow the book verbatim. But it is a nice reference to read along with the course. Research articles are assigned along the way.

## Lecture structure
Our lectures will generally follow this structure, which is roughly equally divided:  
**Part 1:** Followup from last time + Theoretical introduction  
**Part 2:** Practical examples  
**Part 3:** Coding challenge (challenge based on today's lecture)

## Tools and AI
- **Everything in this course runs on free tools:** open-weight models on your own laptop or on [Google Colab](https://colab.research.google.com/) (free GPU). You will never need to pay for anything to solve the coding challenges or the exam.
- **Generative AI and coding agents:** Use them, cite them and check what they produce. SDU allows generative AI in written take-home exams if you declare how you used it ([SDU's guidance](https://sdunet.dk/en/undervisning-og-eksamen/uddannelse/vejledning)). In class, I will regularly show how I work with coding agents.

## Course plan and slides
*Preliminary: this plan will be revised along the way. Slides and coding challenges are linked when they are ready.*

| Kolding | Odense | Slides | Topic | Assigned Reading | Coding challenge |
| --- | --- | --- | --- | --- | --- |
| *Spring* | *Spring* | [Lecture 0 - Introducing the course](https://christianvedels.github.io/News_and_Market_Sentiment_Analytics/Lecture%200%20-%20Introducing%20the%20course/Slides.html) | Motivation for the course, structure and expectations. *Given as an introduction to the course in late spring.* | | |
| 2026-10-26 | 2026-10-28 | [Lecture 1 - Introduction](https://christianvedels.github.io/News_and_Market_Sentiment_Analytics/Lecture%201%20-%20Introduction/Slides.html) | Introduction and the state of NLP: A broad overview, where the field is now, and setting up the course | See below | [Coding challenge 1](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/main/Lecture%201%20-%20Introduction/Coding_challenge1.md) |
| 2026-11-02 | 2026-11-06 | Lecture 2 - Data wrangling with text | Classical data wrangling with text | [Text Wrangling with Python](https://blog.devgenius.io/text-wrangling-with-python-a-comprehensive-guide-to-nlp-and-nltk-f7553e713291) + ['The Zipf Mystery'](https://youtu.be/fCn8zs912OE?si=xVMA63kt9M99Qvjx) | Coding challenge 2 |
| 2026-11-09 | 2026-11-13 | Lecture 3 - Classification | Classification - recipes for training a text classifier, including using LLMs to label data. I will also introduce an example from my own research, OccCANINE. | Ch 2, [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604) | Coding challenge 3 |
| 2026-11-16 | 2026-11-20 | Lecture 4 - Tokenization and Embeddings | Tokenization and Embeddings | Ch 4, Ch 5, [3Blue1Brown: Attention in transformers, visually explained](https://youtu.be/eMlx5fFNoYc?si=HtfNuNlO4Y0e3olH) | Coding challenge 4 |
| 2026-11-23 | 2026-11-27 | Lecture 5 - Causality | A primer on applied causal inference in news and market sentiment analytics (and the productivity of AI) | *See lecture notes* | Coding challenge 5 |
| 2026-11-30 | 2026-12-04 | Lecture 6 - Linking and extracting | From text to tables: record linking, structured extraction and topic modelling | Ch. 8; [Blogpost: Record Linkage with Machine Learning](https://blog.damavis.com/en/record-linkage-with-machine-learning/); [Automated Linking of Historical Data](https://www.nber.org/papers/w25825); [Topic modelling with LDA](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2) | Coding challenge 6 |
| 2026-12-07 | 2026-12-11 | Lecture 7 - From RAG to agents | Retrieval, tool use and agentic workflows for news and market sentiment | [Enhancing Financial Sentiment Analysis via Retrieval Augmented Large Language Models](https://arxiv.org/abs/2310.04027); [Instruct-FinGPT: Financial Sentiment Analysis by Instruction Tuning of General-Purpose Large Language Models](https://arxiv.org/abs/2306.12659) | |
| 2026-12-14 | 2026-12-18 | | Exam workshop | | |

## Exam
One-week take-home project in December. Dates follow. The format will be similar to [last year's](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/blob/old2025/Exam/Format_of_the_exam.md).

## Reading for lecture 1
For the first lecture you have three tasks to do beforehand:

1. Please read Chapter 1 of Patel & Arasanipalai (2021), which is freely available here: https://www.oreilly.com/library/view/applied-natural-language/9781492062561/ch01.html

2. Is ChatGPT high-tech plagiarism? https://youtu.be/SJi4VE-0MoA?si=sqFxbEGQ27drkvFA  

3. This course teaches NLP techniques applied to financial analysis. We can put more or less weight on the application. Please think about the type of skills you want to learn from this course. Then we will adapt the lectures accordingly.

## Previous years
The material from earlier versions of the course is kept on separate branches: [2025](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/tree/old2025), [2024](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/tree/old2024), [2023](https://github.com/christianvedels/News_and_Market_Sentiment_Analytics/tree/old2023).

## Contact information
**The course is taught by:**  
*Christian Vedel,*  
*christian-vs@sam.sdu.dk,*  
*Assistant Professor of Economics,*  
*Department of Economics*  
