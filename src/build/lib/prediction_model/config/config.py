import pathlib
import os
import prediction_model

# root directory of the package
#PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Path to the datasets
DATAPATH = os.path.join(PACKAGE_ROOT, "datasets") 

# datasets
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# traned model path
MODEL_NAME = "classification.pkl"
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, "trained_models")

# features
TARGET = "Survived"
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Fare', 'Embarked'] # todo : remove Fare and Embarked

# remove extra features
DROP_FEATURES = ['Name','PassengerId','Ticket']

# numerical variables 
NUMERICAL_FEATURES = ['Age', 'Fare']

# Categorical variables
CATEGORICAL_FEATURES = ['Sex', 'Embarked', 'Cabin']

# features to encode to ordinal
FEATURES_TO_ORDINAL_ENCODE = ['Cabin', 'Embarked']

# features to encode to one hot
FEATURES_TO_ONEHOTENCODER = ['Sex']

# feature to slice
FEATURES_TO_SLICE = ['Cabin']

# features to modify
FEATURES_TO_MODIFY = ''
FEATURES_TO_ADD = ''


# variables to log transform
LOG_FEATURES = []
# variables to impute

# variables to scale
SCALER_FEATURES = ['Age', 'Fare']