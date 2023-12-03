# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:23:59 2023

@author: chris
"""

from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

token = "YOUR TOKEN GOES HERE" # Get it here: https://huggingface.co/settings/tokens

tokenizer = AutoTokenizer.from_pretrained(model, token = token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    token=token
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")