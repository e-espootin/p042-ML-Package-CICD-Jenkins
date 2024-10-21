import pandas as pd
import numpy as np
import joblib
import logging
import sys
import os
from pathlib import Path
# Add the path to the root directory of the package
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
#
from prediction_model.config import config  
from prediction_model.processing.data_handling import load_pipeline, load_dataset
from prediction_model import pipeline as pipe

classification_pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)

def generate_predictions(*, input_data: pd.DataFrame) -> pd.DataFrame:
    pred = classification_pipeline.predict(input_data[config.FEATURES]) 
    output = np.where(pred==1, 'Y', 'N')
    result = {'prediction': output} 
    return result

# def generate_predictions() -> pd.DataFrame:
#     try:
#         test_data = load_dataset(file_name=config.TEST_FILE)
#         pred = classification_pipeline.predict(test_data[config.FEATURES]) 
#         output = np.where(pred==1, 'Y', 'N')
#         #result = {'predictions': output} 
#         print(output)
#         return output
#     except Exception as e:
#         logging.error(f"Error during generate_predictions: {str(e)}")

if __name__ == '__main__':
    generate_predictions()