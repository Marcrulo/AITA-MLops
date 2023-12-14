import transformers
from datasets import Dataset, load_dataset

import pandas as pd
import os
import glob
import logging

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from model import ModelClass

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
file_handler = logging.FileHandler('make_dataset.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




'''
RAW DATA
'''
try:
    # Check if path exists
    path_raw = 'data/raw/'
    if not os.path.exists(path_raw):
        # Create folder if it does not exist
        logger.info(f'Path "{path_raw}" does not exist. \n[Creating path "{path_raw}"]')
        os.makedirs(path_raw)
        

    # Check if file exists
    file = 'AITA-Reddit-Dataset.csv'
    if os.path.exists(path_raw+file):
        logger.info(f'Path "{path_raw+file}" already exists. \n[Skipping downloading raw dataset]')

    # Load and save raw dataset
    else:
        logger.info("Loading raw dataset...")
        dataset = load_dataset("OsamaBsher/AITA-Reddit-Dataset")
        train_dataset = dataset['train']
        train_df = train_dataset.to_pandas()
        logger.info("Saving raw dataset...")
        train_df.to_csv(path_raw+file, index=False)
        del train_df, train_dataset, dataset
        logger.info("Dataset saved!")

except Exception as e:
    logger.exception(e)

'''
PROCESSED DATA
'''

try: 
    # Check if path exists
    path_processed = 'data/processed/'
    if not os.path.exists(path_processed):
        logger.info(f'Path "{path_processed}" does not exist. \n[Creating path "{path_processed}"]')
        os.makedirs(path_processed)

    # Check if file exists
    file = 'AITA-Reddit-Dataset.csv'
    if os.path.exists(path_processed+file):
        logger.info("Processed dataset already exists. \n[Skipping processing data]")
    else:
        logger.info("Processing data...")

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
        logger.info("Processing done!")
        
except Exception as e:
    logger.exception(e)   


'''
TOKENIZED DATA
'''    
try: 
    # Check if path exists
    path_tokenized = 'data/tokenized/'
    if not os.path.exists(path_tokenized):
        logger.info(f'Path "{path_tokenized}" does not exist. \n[Creating path "{path_tokenized}"]')
        os.makedirs(path_tokenized)

    # Check if file exists
    file = 'dataset.hf'
    if os.path.exists(path_tokenized+file):
        logger.info("Tokenized dataset already exists. \n[Skipping tokenizing data]")
    else:
        logger.info("Tokenizing data...")

        # Read checkpoint
        checkpoint = glob.glob(os.path.join('results','*'))
        checkpoint.sort(key=os.path.getmtime)
        checkpoint.insert(0,None) # final checkpoint if None, if no actual checkpoints are present
        chkpt = checkpoint[-1]

        # Load tokenizer
        Model = ModelClass(chkpt)
        tokenizer = Model.load_tokenizer()

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
        dataset.save_to_disk(f'data/tokenized/{file}')
        
except Exception as e:
    logger.exception(e)