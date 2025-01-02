
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_pipeline
from prediction_model.processing import preprocessing as pp
from prediction_model import pipeline as pipe
from prediction_model.load_data.load import ClickhouseDataLoader
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

#
# PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
# PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# sys.path.append(str(PACKAGE_ROOT))


# Perform the training
def perform_training():
    try:
        # load data into dataset
        '''
        loader = ClickhouseDataLoader()
        loaded_df = loader.load_data('SELECT * FROM testdb1.fin_transactions;')
        # print(loaded_df.head())
        train_df, test_df = loader.split_data(
            loaded_df, test_size=0.2, random_state=42)
        loader.save_dataframes(train_df, test_df, config.DATAPATH)
        '''
        # Load the dataset
        data = load_dataset(file_name=config.TRAIN_FILE)

        # Split the dataset
        X_train = data[config.FEATURES]
        y_train = data[config.TARGET]  # .map({'N':0, 'Y':1})

        logging.info(f'Starting training with data: {config.FEATURES}')
        # Fit the pipeline
        pipe.classification_pipeline.fit(X_train[config.FEATURES], y_train)

        # Save the pipeline
        save_pipeline(pipeline_to_persist=pipe.classification_pipeline)
    except Exception as e:
        logging.error(f"Error during perform_training: {str(e)}")

# evaluate the performance of model with metrics


if __name__ == '__main__':
    print('Training pipeline')
    perform_training()
