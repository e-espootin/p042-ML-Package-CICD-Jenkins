import os
import pandas as pd
import joblib
from prediction_model.config import config

# Load the dataset
def load_dataset(*, file_name: str) -> pd.DataFrame:
    #_data = pd.read_csv(f"{config.DATAPATH}/{file_name}")
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data

# Serializing the model
def save_pipeline(*, pipeline_to_persist) -> None:
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_persist, save_path)
    print(f"saved Model: {config.MODEL_NAME}")

# Deserializing the model
def load_pipeline(*, pipeline_to_load: str):
    file_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    trained_model = joblib.load(filename=file_path)
    print(f"loaded Model: {config.MODEL_NAME}")
    return trained_model

