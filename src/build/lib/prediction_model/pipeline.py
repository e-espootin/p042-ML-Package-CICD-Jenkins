from sklearn.pipeline import Pipeline   
import numpy as np
import sys
import os
from pathlib import Path
# Add the path to the root directory of the package
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
#
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


classification_pipeline = Pipeline(
    [
        ('mean_imputer', pp.MeanImputer(variables=config.NUMERICAL_FEATURES)),
        ('mode_imputer', pp.ModeImputer(variables=config.CATEGORICAL_FEATURES)),
        ('cut_a_slice', pp.CutASlice(variables=config.FEATURES_TO_SLICE, start=0, end=1)),
        ('LabelEncoder', pp.CutomeLabelEncoder(variables=config.FEATURES_TO_ORDINAL_ENCODE)),
        ('OneHotEncoder', pp.CutomeOneHotEncoder(variables=config.FEATURES_TO_ONEHOTENCODER)),
        #('drop_columns', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)), # already filtered with config.FEATURES
        ('MinMaxScaler', MinMaxScaler()),
        ('RandomForestClassifier', RandomForestClassifier(random_state=0)),
    ]

)