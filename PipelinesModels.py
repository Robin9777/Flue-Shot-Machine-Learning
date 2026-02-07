from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from config import PARAMS_DT, PARAMS_RF
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import make_column_selector
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import logging


# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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



# ========= Abstract Base Class for Model Pipelines =========
class ModelPipeline(ABC):

    @abstractmethod
    def build_pipeline(self, df_: pd.DataFrame, drop_missing_threshold=0.2) -> Pipeline:
        pass


class PipelineFactory:

    @staticmethod
    def get_pipeline(model_type: str) -> ModelPipeline:
        if model_type == 'decision_tree':
            return DTPipeline()
        elif model_type == 'random_forest':
            return RFPipeline()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        

# ========= Decision Tree Pipeline Implementation =========
class DTPipeline(ModelPipeline):

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
            n_jobs=-1
        )
 
        return Pipeline([
            ('drop_missing', DropMissingTransformer(threshold=drop_missing_threshold)),
            ('column_transformer', column_transformer),
            ('selectkbest', SelectKBest(score_func=chi2)),
            ('dt', model)
        ])
    


# ========= Random Forest Pipeline Implementation =========
class RFPipeline(ModelPipeline):

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

        model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )

        return Pipeline([
            ('drop_missing', DropMissingTransformer(threshold=drop_missing_threshold)),
            ('column_transformer', column_transformer),
            ('selectkbest', SelectKBest(score_func=chi2)),
            ('rf', model)
        ])



# ========= Flue Shot Model Class for Training and Evaluation =========
class FlueShotModel():

    PARAMS_DT = PARAMS_DT
    PARAMS_RF = PARAMS_RF
    drop_missing_threshold = 0.2

    def __init__(
            self, 
            model_type: str,
            df_x: pd.DataFrame,
            df_y: pd.DataFrame
    ):
        self.pipeline_model = PipelineFactory.get_pipeline(model_type)
        self.df_x = df_x
        self.df_y = df_y
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()

        # Caching
        self._best_models = {}
        self.grid_searchs = {}


    def _split_data(self, test_size=0.2, random_state=42):
        return train_test_split(self.df_x, self.df_y, test_size=test_size, random_state=random_state)
    

    def _pipeline_builder(self):
        return self.pipeline_model.build_pipeline(self.df_x, drop_missing_threshold=self.drop_missing_threshold)
    
    def _grid_search(self, cv=5, scoring='roc_auc'):

        pipeline = self._pipeline_builder()

        logging.info(f"Pipeline built with steps: {pipeline.named_steps.keys()}")

        if isinstance(pipeline.named_steps['rf'], RandomForestClassifier):
            param_grid = self.PARAMS_RF
        else:
            param_grid = self.PARAMS_DT

        return GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
    

    def _get_roc_auc_score(self, model, label, set_="test"):
        if set_ == "train":
            y_pred_proba = model.predict_proba(self.X_train)[:, 1]
            return roc_auc_score(self.y_train[label], y_pred_proba)
        else:
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            return roc_auc_score(self.y_test[label], y_pred_proba)
    

    def fit_grid_search(self, label):

        # looking in cache
        if label in self.grid_searchs:
            logging.info(f"Grid search for label '{label}' found in cache.")
            return self.grid_searchs[label]
        # not found in cache, fit and store
        logging.info(f"Fitting grid search for label '{label}'...")
        grid_search = self._grid_search()
        grid_search.fit(self.X_train, self.y_train[label])
        self.grid_searchs[label] = grid_search
        return grid_search
    
    
    def get_model(self, label):

        # looking in cache
        if label in self._best_models:
            logging.info(f"Best model for label '{label}' found in cache.")
            return self._best_models[label]
        # not found, fit and store
        logging.info(f"Getting best model for label '{label}' by fitting grid search...")
        gs = self.fit_grid_search(label)
        self._best_models[label] = gs.best_estimator_
        return gs.best_estimator_
    

    def print_classification_report(self, label, set_="test"):
        model = self.get_model(label)
        if set_ == "train":
            y_pred = model.predict(self.X_train)
        else:
            y_pred = model.predict(self.X_test)

        print(f"\nClassification Report for {set_} set")
        print(classification_report(self.y_test[label], y_pred))


    def print_roc_auc_score(self, label, set_="test"):
        model = self.get_model(label)
        score = self._get_roc_auc_score(model, label, set_=set_)
        print(f"\nROC AUC Score for {set_} set: {score:.4f}")


    def prepare_submission(self, test_df:pd.DataFrame, output_file="submission.csv"):
        model_h1n1 = self.get_model("h1n1_vaccine")
        model_seasonal = self.get_model("seasonal_vaccine")
        predictions_h1n1 = model_h1n1.predict_proba(test_df)[:, 1]
        predictions_seasonal = model_seasonal.predict_proba(test_df)[:, 1]
        submission = pd.DataFrame(
            index=test_df.index, columns=['h1n1_vaccine', 'seasonal_vaccine']
        )
        submission['h1n1_vaccine'] = predictions_h1n1
        submission['seasonal_vaccine'] = predictions_seasonal
        submission.to_csv(output_file, index_label='respondent_id')





if __name__ == "__main__":

    df = pd.read_csv("training_set_features.csv", index_col="respondent_id")
    results = pd.read_csv("training_set_labels.csv", index_col="respondent_id")

    model_builder = FlueShotModel("random_forest", df, results)

    df_test = pd.read_csv("test_set_features.csv", index_col="respondent_id")

    model_builder.prepare_submission(df_test, output_file="submission.csv")
    logging.info("Model obtained successfully.")