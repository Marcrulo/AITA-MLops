import transformers
from datasets import Dataset, load_dataset, load_from_disk
import evaluate

import pandas as pd
import torch
import logging
import glob
import os

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import ModelClass

# load data
dataset = load_from_disk('data/tokenized/dataset.hf')
# print("Dataframe loaded")
# dataset = Dataset.from_pandas(df)

# return latest checkpoint - or None, if no checkpoint exists
checkpoint = glob.glob(os.path.join('results','*'))
checkpoint.sort(key=os.path.getmtime)
checkpoint.insert(0,None) # final checkpoint if None, if no actual checkpoints are present
chkpt = checkpoint[-1]

# load model
Model = ModelClass(chkpt)
model = Model.load_model()
tokenizer = Model.load_tokenizer()

# metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels)

# fine-tune model
training_args = transformers.TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=1,   # batch size per device during training
    per_device_eval_batch_size=1,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    evaluation_strategy='steps',
    logging_dir='./logs',            # directory for storing logs
    eval_steps=500,
    save_steps=500,
)
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)
if chkpt is not None:
    trainer.train(chkpt)
else:
    trainer.train()