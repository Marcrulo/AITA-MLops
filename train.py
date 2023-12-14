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

# return latest checkpoint - or None, if no checkpoint exists
checkpoint = glob.glob(os.path.join('results','*'))
checkpoint.sort(key=os.path.getmtime)
checkpoint.insert(0,None) # final checkpoint if None, if no actual checkpoints are present
chkpt = checkpoint[-1]
    

# load model
Model = ModelClass(chkpt)
model = Model.load_model()
tokenizer = Model.load_tokenizer()


