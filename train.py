import transformers
from datasets import Dataset, load_dataset
import evaluate

import pandas as pd
import torch
import logging
import glob
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# load data
df = load_dataset('tokenized/dataset.hf')

# get last created file
checkpoint = glob.glob(os.path.join('results','*'))
checkpoint.sort(key=os.path.getmtime)

# return latest checkpoint - or None, if no checkpoint exists
if len(checkpoint) > 0:
    chkpt = checkpoint[-1]
else:
    chkpt = None
    

# load model
model, tokenizer = LoadModel(chkpt)


