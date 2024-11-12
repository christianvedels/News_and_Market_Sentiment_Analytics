# Week 3 Coding Challenge: Emotions from tweets

## Challenge Details
- Run the following simple ZeroShotClassification:

```
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model = "typeform/distilbert-base-uncased-mnli"
                      )
candidate_labels = ["Joy", "Anger", "Surprise", "Sadness", "Fear", "Confidence"]

classifier(
    "And then something dreadful happenened. I wouldn't dare to rethink it if I had no need.",
    candidate_labels
    )
```

- Wrap this in a function GetEmotions(x)

- Apply the function to 20 positive and 20 negative tweets 

- Make an illustration of the average emotions in negative and positive tweets

## Resources

**Hint 1:** You need to install NLTK Data which contains a library of tweets  
See here: https://www.nltk.org/install.html 

**Hint 2:** This is how you load the tweets
```
import nltk
from nltk.corpus import twitter_samples

nltk.download("twitter_samples")
tweets_neg = twitter_samples.strings("negative_tweets.json")
tweets_pos = twitter_samples.strings("positive_tweets.json")

[Remember to then also take a random subset of 20 from each]
```

**Hint 3:**
It is easier to discern differences if you subtract out the mean
```
# Subtract mean from tweet emotions
mean_emotions = (average_emotions_neg + average_emotions_pos)/2
average_emotions_neg = average_emotions_neg - mean_emotions
average_emotions_pos = average_emotions_pos - mean_emotions
```
