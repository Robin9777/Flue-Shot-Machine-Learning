import json
from sklearn.base import BaseEstimator, TransformerMixin



def read_json(file_name:str):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data



class DropMissingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.cols_to_keep_ = None

    def fit(self, X, y=None):
        missing_ratio = X.isna().sum() / len(X)
        self.cols_to_keep_ = missing_ratio[missing_ratio <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X[self.cols_to_keep_]