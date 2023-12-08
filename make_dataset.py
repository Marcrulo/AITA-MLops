from datasets import load_dataset
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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
print("Done!")
    
    
    
# Tokenize and format data
...