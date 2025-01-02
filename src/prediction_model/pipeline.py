from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from prediction_model.processing import preprocessing as pp
from prediction_model.config import config
from sklearn.pipeline import Pipeline
import numpy as np
import sys
import os
from pathlib import Path
# Add the path to the root directory of the package
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
#

# TODO :
# add oversampling > SMOTE
# add Undersampling > RandomUnderSampler
classification_pipeline = Pipeline(
    [
        ('mean_imputer', pp.MeanImputer(variables=config.NUMERICAL_FEATURES)),
        ('mode_imputer', pp.ModeImputer(variables=config.CATEGORICAL_FEATURES)),
        # ('cut_a_slice', pp.CutASlice(variables=config.FEATURES_TO_SLICE, start=0, end=1)),
        ('LabelEncoder', pp.CutomeLabelEncoder(
            variables=config.FEATURES_TO_ORDINAL_ENCODE)),
        ('OneHotEncoder', pp.CutomeOneHotEncoder(
            variables=config.FEATURES_TO_ONEHOTENCODER)),
        # already filtered with config.FEATURES
        ('drop_columns', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        # imbalanced > oversampling
        # ('smote', SMOTE(random_state=42)),
        # ('model', RandomForestClassifier(class_weight='balanced', random_state=42))
        #
        ('MinMaxScaler', MinMaxScaler()),
        ('RandomForestClassifier', RandomForestClassifier(random_state=0)),
    ]

)
