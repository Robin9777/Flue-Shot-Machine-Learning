from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import utilsfunc as uf
from config import SELECTED_FEATURES, TARGETS, MAPPING
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import pickle


# Missing Values Transformer
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
    


class DTPipeline:

    MAPPING = MAPPING
    TARGETS = TARGETS
    SELECTED_FEATURES = SELECTED_FEATURES

    def __init__(self):
        self.pipeline = None


    def build_pipeline(self, df_: pd.DataFrame, drop_missing_threshold=0.2)-> Pipeline:

        df = df_.copy()

        for col, mapping in self.MAPPING.items():
            df[col] = df[col].map(mapping)

        dp_transformer = DropMissingTransformer(threshold=drop_missing_threshold)

        Column_transformer = ColumnTransformer(
            transformers=[
                ("imputer", SimpleImputer(strategy="median"), SELECTED_FEATURES)
            ], 
            remainder="passthrough"
        )

        return Pipeline([
            ('drop_missing', uf.DropMissingTransformer(threshold=0.2)),
            ('column_transformer', Column_transformer),
            ('selectkbest', SelectKBest(score_func=chi2, k=10)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ])
    
    
    

        



