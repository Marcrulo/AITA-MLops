import transformers
from datasets import Dataset, load_dataset

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import pandas as pd
import os
import glob
import logging

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
file_handler = logging.FileHandler('data/make_dataset.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

from models.model import ModelClass




'''
GCP BUCKET
'''
from google.cloud import storage

# Create a storage client
storage_client = storage.Client()

# Specify the bucket and blob (file) names
bucket_name = 'aita_datasets'

# Get a reference to the bucket
bucket = storage_client.bucket(bucket_name)


def exists_in_gcs(object_name):
    blob = bucket.blob(object_name)
    return blob.exists()



'''
RAW DATA
'''
try:
    # Check if file exists
    path_raw = 'data/raw/'   
    file = 'AITA-Reddit-Dataset.csv'
    if exists_in_gcs(path_raw+file):
        logger.info(f'Path "{path_raw+file}" already exists. \n[Skipping downloading raw dataset]')

    # Load and save raw dataset
    else:
        os.makedirs(path_raw)
        logger.info("Loading raw dataset...")
        dataset = load_dataset("OsamaBsher/AITA-Reddit-Dataset")
        train_dataset = dataset['train']
        train_df = train_dataset.to_pandas()
        logger.info("Saving raw dataset...")
        train_df.to_csv(path_raw+file, index=False)
        logger.info("Dataset saved!")
        del train_df, train_dataset, dataset
        blob = bucket.blob(path_raw+file)
        blob.upload_from_filename(path_raw+file)
        logger.info(f'File {file} uploaded to gs://{bucket_name}/{blob_name}')

except Exception as e:
    logger.exception(e)
    assert "Error with downloading raw dataset"




'''
PROCESSED DATA
'''

try: 

    # Check if file exists
    path_processed = 'data/processed/'
    file = 'AITA-Reddit-Dataset.csv'
    if exists_in_gcs(path_processed+file):
        logger.info("Processed dataset already exists. \n[Skipping processing data]")
    
    else:
        os.makedirs(path_processed)
        logger.info("Processing data...")

        # Process data
        #df = pd.read_csv(path_raw+file)
        blob = bucket.blob(path_raw+file)
        file_contents = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(file_contents))
        
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
        
        blob = bucket.blob(path_processed+file)
        blob.upload_from_filename(path_processed+file)
        logger.info(f'File {file} uploaded to gs://{bucket_name}/{blob_name}')
        
except Exception as e:
    logger.exception(e)
    assert "Error with downloading processed dataset"   


'''
TOKENIZED DATA
'''    
try: 
    # Check if file exists
    path_tokenized = 'data/tokenized/'
    file = 'dataset.hf'
    if exists_in_gcs(path_tokenized+file):
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
        # df = pd.read_csv(path_processed+file)
        blob = bucket.blob(path_processed+file)
        file_contents = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(file_contents))
        
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
        dataset.save_to_disk(path_tokenized+file)
        
        blob = bucket.blob(path_tokenized+file)
        blob.upload_from_filename(path_tokenized+file)
        logger.info(f'File {file} uploaded to gs://{bucket_name}/{blob_name}')
        
except Exception as e:
    logger.exception(e)
    assert "Error with tokenizing dataset"