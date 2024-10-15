

# Week 1 Coding Challenge: Skipping ahead to the end

### 1. Your computer can read

- Use transformers to run a sentiment analysis on a sentence in less than 20 lines of code.
- Try it on this sentence: "That would have been splendid. Absolutely amazing. But it was quite the opposite."
- Would you think the sentence above would be classified as positive or negative based on the words alone without the context of the entire sentence?

See this: [Getting Started with Hugging Face](https://www.kaggle.com/code/anubhavgoyal10/getting-started-with-hugging-face)

Here is something to get you started:

```python
from transformers import pipeline
classifier = pipeline([insert task name])
```

### 2. Your computer can easily read an entire book
- Build a function "GetSentiment" which should return the sentiment of a list of sentences (based on the approach above).

- Run this function on every single sentence of "Notes from the Underground".

- Visualize how the sentiment of "Notes from the Underground" evolves through the entire book. (It might be handy to estimate a moving average of e.g. 50 sentences to 'smooth' out the sentence by sentence sentiment.)

Resources: ["Notes from the Underground" by Fyodor Dostoyevsky](https://www.gutenberg.org/cache/epub/600/pg600.txt)

**Extra:** You can also try with ["Pride and Prejudice" by Jane Austen](https://www.gutenberg.org/files/1342/1342-0.txt)
#### Hint 1
You can steal this function to load up the the entire text
```python
def get_notes_from_the_underground():
    """
    This function retrieves *Notes from the Underground* from the Gutenberg
    website. It also does some text cleaning and separates each sentence into
    list elements.
    """
    url = 'https://www.gutenberg.org/cache/epub/600/pg600.txt'
    response = requests.get(url)
    if response.status_code == 200:
        response.encoding = 'utf-8'
        text = response.text

        # Split text into sentences
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\r\n\r\n', text)
        # Clean text
        sentences = sentences[11:]
        sentences = sentences[:-117]
        return sentences
    else:
        raise Exception(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
    return text
```

#### Hint 2
The following function might also be handy:
```python
def get_positive_score(x):
    """
    The pipeline returns 'POSITIVE' or 'NEGATIVE' and a probability, where the
    label is based on what is the most likely sentiment of the sentence. It
    turns out to be useful to have one continuous score from -1 to 1, which
    captures completely 'postive' if 1 and completely 'negative' if -1. This
    function handles that.
    """
    if x["label"] == "POSITIVE":
        res_x = x['score']
    elif x["label"] == "NEGATIVE":
        res_x = 1 - x['score']
    else:
        raise Exception(x["label"]+"This should not be possible")

    res_x = res_x*2-1 # Expand to -1 to 1 scale

    return res_x
```
