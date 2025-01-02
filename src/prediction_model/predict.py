from prediction_model.config import config
from prediction_model import pipeline as pipe
from prediction_model.processing.data_handling import load_pipeline, load_dataset
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

classification_pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)


def generate_predictions(*, input_data: pd.DataFrame) -> pd.DataFrame:
    pred = classification_pipeline.predict(input_data[config.FEATURES])
    output = np.where(pred == 1, 'Y', 'N')
    result = {'prediction': output}
    return result


def generate_predictions_on_test() -> pd.DataFrame:
    try:
        test_data = load_dataset(file_name=config.TEST_FILE)
        pred = classification_pipeline.predict(test_data[config.FEATURES])
        output = np.where(pred == 1, 'Y', 'N')
        # result = {'predictions': output}
        print(output)
        return output
    except Exception as e:
        logging.error(f"Error during generate_predictions: {str(e)}")

# # TODO : make evaluation function
# def evaluate_model(model: ):
#       prediction = classification_pipeline.predict(df)
#     y_pred = model.predict(X_test)
#     y_proba = model.predict_proba(X_test)[:, 1]  # For ROC-AUC

#     print("Precision:", precision_score(y_test, y_pred))
#     print("Recall:", recall_score(y_test, y_pred))
#     print("F1-score:", f1_score(y_test, y_pred))
#     print("ROC-AUC:", roc_auc_score(y_test, y_proba))


if __name__ == '__main__':
    generate_predictions()
