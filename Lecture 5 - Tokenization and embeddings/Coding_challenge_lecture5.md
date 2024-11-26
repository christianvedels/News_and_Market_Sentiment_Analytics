# Week 5 Embeddings and zero shot classifiers


1. Get the text embeddings from `BERT`
1. Build a zero shot classifier using cosine distance. It should be a 

```python 
model = zeroshot()
``` 


```python
from transformers import pipeline

extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction")
result = extractor("This is a simple test.", return_tensors=True)
result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension] representing the input string.
```