from sklearn.base import BaseEstimator, TransformerMixin
from prediction_model.config import config
import numpy as np

# Mean imputer

class MeanImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.mean_dict = X[self.variables].mean().to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.mean_dict[feature])
        return X

# Mode imputer    
class ModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.mode_dict = X[self.variables].mode().to_dict()
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].fillna(self.mode_dict[feature])
        return X
    

# Drop columns    
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables = variables_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(columns = self.variables, axis= 1)
        return X
    
# Cut a slice of a string
class CutASlice(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, start=None, end=None):
        self.variables = variables
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].str[self.start:self.end]
        return X

# combine two columns
class DomainProcessing(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_modify=None, variables_to_add=None):
        self.variables_to_modify = variables_to_modify
        self.variables_to_add = variables_to_add

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X[self.variables_to_modify] = X[self.variables_to_modify] + X[self.variables_to_add]
        return X
    
# label encoding
class CutomeLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables


    def fit(self, X, y=None):
        self.label_dict = {}
        for feature in self.variables:
            t = X[feature].value_counts().sort_values(ascending=True).index
            self.label_dict[feature] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.label_dict[feature])
        return X
    

# one hot encoding
class CutomeOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self.label_dict = {}
        for feature in self.variables:
            t = X[feature].value_counts().sort_values(ascending=True).index
            self.label_dict[feature] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            for label in self.label_dict[feature].keys():
                X[feature+'_'+str(label)] = np.where(X[feature]==label, 1, 0)
            X = X.drop(columns=[feature])
        return X
    
# try out log transformation
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = np.log(X[feature])
        return X