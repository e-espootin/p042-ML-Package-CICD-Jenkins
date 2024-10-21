import pytest

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.processing import preprocessing as pp
from prediction_model.predict import generate_predictions

@pytest.fixture()
def single_prediction():
    # Given
    test_data = load_dataset(file_name=config.TEST_FILE)
    single_row = test_data[:1]
    # When
    result = generate_predictions(input_data=single_row)
    
    return result

def test_single_pred_not_none(single_prediction): # output from predict script not null
    assert single_prediction is not None

def test_single_pred_data_type(single_prediction): # data type is str
    assert isinstance(single_prediction.get('prediction')[0], str)

def test_single_pred_validate(single_prediction): # output is Y for an example data
    assert single_prediction.get('prediction')[0] == 'N'