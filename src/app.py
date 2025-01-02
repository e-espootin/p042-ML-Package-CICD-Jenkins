# test purpose
from prediction_model.processing import data_handling
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, load_pipeline
from prediction_model.predict import generate_predictions
from prediction_model import training_pipeline
import pandas as pd
from prediction_model.processing import preprocessing as pp
#
from pathlib import Path
import sys
# load the model
classification_pipeline = load_pipeline(pipeline_to_load=config.MODEL_NAME)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PACKAGE_ROOT))


def main():
    # a = pp.CustomLabelEncoder_v2()

    print('Training pipeline')
    print(f"PACKAGE_ROOT: {PACKAGE_ROOT}")

    # 100639,2025-01-01 21:15:07,781214.63,"Taylor, Howell and Jackson",120,Derrick Taylor,549,259,VISA 13 digit,361,Credit,Discover,Purchase,Failed,Retail,Online,Bank of America,Citi,0
    # new_data = {}
    test_data = load_dataset(file_name=config.TEST_FILE)
    # single_row = test_data[:1] # random row
    single_row = test_data[test_data['fraud'] == 1][:1]
    print(single_row.head())
    print(f"amount: {single_row['amount']}")
    print(f"location_id: {single_row['location_id']}")
    df = pd.DataFrame(single_row[config.FEATURES])
    # fit_transform > get transformed dataset
    # prediction = classification_pipeline.fit_transform(df)

    prediction = classification_pipeline.predict(df)
    # gen predictions
    # prediction = generate_predictions(input_data=df)
    # prediction = training_pipeline.perform_training()

    print(f'Prediction: {prediction}')


if __name__ == '__main__':
    main()
