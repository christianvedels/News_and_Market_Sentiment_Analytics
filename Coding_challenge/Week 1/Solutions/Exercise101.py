# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:02:17 2023

@author: chris
"""
#%%
# Set path to file path
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
#%%

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

classifier("That would have been splendid. Absoloutly amazing. But it was quite the opposite.")


