import pathlib
import os
import prediction_model

# root directory of the package
# PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
PACKAGE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir))

# Path to the datasets
DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

# datasets
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# traned model path
MODEL_NAME = "classification.pkl"
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")

# features
TARGET = "fraud"
FEATURES = [
    'transaction_id', 't_datetime', 'amount', 'merchant_name', 'merchant_id',
    'customer_name', 'customer_id', 'location_id', 'payment_method', 'terminal_id',
    'card_type', 'card_brand', 'transaction_type', 'transaction_status',
    'transaction_category', 'transaction_channel', 'merchant_bank', 'customer_bank'
]

# remove extra features
DROP_FEATURES = ['transaction_id', 't_datetime',
                 'merchant_name', 'customer_name']

# numerical variables
NUMERICAL_FEATURES = ['amount']

# Categorical variables
CATEGORICAL_FEATURES = ['merchant_id', 'customer_id',
                        'location_id', 'payment_method', 'terminal_id', 'card_brand', 'transaction_type', 'transaction_status',
                        'transaction_category', 'transaction_channel', 'merchant_bank', 'customer_bank']

# features to encode to ordinal
FEATURES_TO_ORDINAL_ENCODE = ['merchant_id', 'customer_id', 'location_id', 'terminal_id',
                              'card_brand',
                              'transaction_category', 'merchant_bank', 'customer_bank', 'transaction_channel', 'payment_method']

# features to encode to one hot
FEATURES_TO_ONEHOTENCODER = ['transaction_status',
                             'transaction_type', 'card_type']
FEATURES_TO_LABEL_ENCODE = []
# feature to slice
FEATURES_TO_SLICE = []

# features to modify
FEATURES_TO_MODIFY = ''
FEATURES_TO_ADD = ''


# variables to log transform
LOG_FEATURES = []
# variables to impute

# variables to scale
SCALER_FEATURES = []
