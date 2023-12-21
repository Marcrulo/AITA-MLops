import transformers
from datasets import load_from_disk

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch
import numpy as np

import glob
import os
from models.model import ModelClass

# Load checkpoint
checkpoint = glob.glob(os.path.join('models/results','*'))
checkpoint.sort(key=os.path.getmtime)
checkpoint.insert(0,None) # final checkpoint if None, if no actual checkpoints are present
chkpt = checkpoint[-1]

# Load model + tokenizer
device = 'cuda'
Model = ModelClass(chkpt)
model = Model.load_model()
model.eval()

# load test data
dataset = load_from_disk('data/tokenized/dataset.hf')['test']

# define data loader
def collate_fn(batch):
    return {
        'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
        'token_type_ids': torch.stack([torch.tensor(item['token_type_ids']) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
        'labels': torch.tensor([item['labels'] for item in batch]),
    }

data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# evaluation
true_labels = []
all_logits = []
for batch in data_loader:
    
    # Move your batch to the same device as your model
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
        # Your outputs include things like loss and logits
        logits = outputs.logits
        
        # Calculate metrics here based on logits and batch['labels']
        true_labels.append(batch['labels'].cpu().numpy())
        all_logits.append(logits.cpu().numpy())

nplogits = np.concatenate(all_logits, axis=0)

predictions = np.argmax(nplogits, axis=1)
true_labels_concat = np.concatenate(true_labels, axis=0).argmax(axis=1)

class_rep = classification_report(true_labels_concat, predictions, output_dict=True)
print(class_rep)