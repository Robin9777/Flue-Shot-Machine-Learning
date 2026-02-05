from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import utilsfunc as uf
from config import SELECTED_FEATURES, TARGETS, MAPPING
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import make_column_selector
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


        cat_pipeline = Pipeline([
            ('impute_cat', SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        num_pipeline = Pipeline([
            ('impute_num', SimpleImputer(strategy='median'))
        ])

        column_transformer = ColumnTransformer(
            transformers=[
                ('cat', cat_pipeline, make_column_selector(dtype_include=object)),
                ('num', num_pipeline, make_column_selector(dtype_include=np.number)),
            ], 
            remainder="drop"
        )

        model = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced',
        )

        return Pipeline([
            ('drop_missing', uf.DropMissingTransformer(threshold=drop_missing_threshold)),
            ('column_transformer', column_transformer),
            ('selectkbest', SelectKBest(score_func=chi2)),
            ('dt', model)
        ])