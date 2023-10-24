

# Week 1 Coding Challenge: Skipping ahead to the end

**Submit here: [Challenge Submission Form](https://forms.gle/WmSEkZn8WH1fiDjE6)**
    
+ Each bullet solved: 5 points  
+ Submitting on time: 10 points  

**Additional requirement: Code must execute out of the box.**


### 1. In 10 minutes

- Use transformers to run a sentiment analysis on a sentence in less than 20 lines of code.
- Try it on this sentence: "That would have been splendid. Absoloutly amazing. But it was quite the opposite."

See this: [Getting Started with Hugging Face](https://www.kaggle.com/code/anubhavgoyal10/getting-started-with-hugging-face)

Here is something to get you started:

```python
from transformers import pipeline
classifier = pipeline([insert task name])
```

### 2. In 25 minutes
- Build a function "GetSentiment" which should return the sentiment of a list of paragraphs (based on the approach above).

- Run this function on every single paragraph of Pride and Prejudice by Jane Austen.

- Visualize how the sentiment of Pride and Prejudice evolves.

Resources: [Pride and Prejudice Text](https://www.gutenberg.org/files/1342/1342-0.txt)
