from datasets import load_dataset
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

'''
RAW DATA
'''

# Check if path exists
path_raw = 'data/raw/'
if not os.path.exists(path_raw):
    # Create folder if it does not exist
    print(f'Path "{path_raw}" does not exist. \n[Creating path "{path_raw}"]')
    os.makedirs(path_raw)
    

# Check if file exists
file = 'AITA-Reddit-Dataset.csv'
if os.path.exists(path_raw+file):
    print(f'Path "{path_raw+file}" already exists. \n[Skipping downloading raw dataset]')

# Load and save raw dataset
else:
    print("Loading raw dataset...")
    dataset = load_dataset("OsamaBsher/AITA-Reddit-Dataset")
    train_dataset = dataset['train']
    train_df = train_dataset.to_pandas()
    print("Saving raw dataset...")
    train_df.to_csv(path_raw+file, index=False)
    del train_df, train_dataset, dataset



'''
PROCESSED DATA
'''

# Check if path exists
path_processed = 'data/processed/'
if not os.path.exists(path_processed):
    print(f'Path "{path_processed}" does not exist. \n[Creating path "{path_processed}"]')
    os.makedirs(path_processed)

# Check if file exists
if os.path.exists(path_processed+file):
    print("Processed dataset already exists. Overwritting...")
else:
    print("Processing data...")

# Process data
df = pd.read_csv(path_raw+file)
df = df[["text", "verdict"]]
preprocessor = ColumnTransformer(
transformers=[
    ('Text', 'passthrough', ['text']),
    ('Verdict', OneHotEncoder(), ['verdict'])
])

processed = preprocessor.fit_transform(df)
df = pd.DataFrame(processed, columns=['text','ESH','NAH','NTA','YTA'])

df.to_csv(path_processed+file, index=False)
del df, processed, preprocessor
print("Done!")
    


'''
TOKENIZED DATA
'''    
    
# Tokenize and format data
model = LoadModel(chkpt)
model_name = model.model_name
tokenizer = transformers.BertTokenizer.from_pretrained(model_name, problem_type='multi_label_classification')

# Create Dataset
df = pd.read_csv('data/processed/AITA-Reddit-Dataset.csv')
base_dataset = Dataset.from_pandas(df)

# Vectorize labels
cols = base_dataset.column_names
base_dataset = base_dataset.map(lambda x: {'labels': [x[c] for c in cols if c != "text"]})

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize the data (only keep relevant columns)
cols = base_dataset.column_names
cols.remove('labels')
dataset = base_dataset.select(range(1000)).map(tokenize_function, batched=True, remove_columns=cols)

# Split dataset
dataset = dataset.train_test_split(test_size=0.2)
validation_test_dataset = dataset['test'].train_test_split(test_size=0.5)
dataset['validation'] = validation_test_dataset['train']
dataset['test'] = validation_test_dataset['test']

# save dataset
dataset.save_to_disk('data/tokenized/dataset.hf')