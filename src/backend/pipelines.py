import os
import sys
import pickle

from typing import List, Tuple, Dict, Any, Optional, Union

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone, check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve

from src.backend.preprocessors import SplitterPreprocessor
from src.backend.evaluation import MetricsEvaluator

import nltk
import spacy

class BasePipelineModule(Pipeline):
    def __init__(self, steps, memory=None, verbose=False, load_path='', save_path=''):
        super().__init__(steps, memory=memory, verbose=verbose)

        self.load_path = load_path
        self.save_path = save_path
    
    def load_from_path(self):
        try:
            with open(self.load_path, 'rb') as f:
                pipeline = pickle.load(f)
        except FileNotFoundError:
            print(f'File not found: {self.load_path}')
            pipeline = None

        return pipeline
    
    def save_to_path(self):
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f)
        except FileNotFoundError:
            print(f'File not found: {self.save_path}')
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return np.zeros(len(X))
    
    def is_fitted(self):
        return check_is_fitted(self)


class PipelineModules(BasePipelineModule):
    def __init__(
        self,
        from_file: bool = False,
        load_path: str = '',
        save_path: str = '',
        steps: Tuple[str, Any] = None,
        memory: str = None,
        device: str = 'cpu',
        verbose: bool = False
    ) -> None:
        super().__init__(steps, memory=memory, verbose=verbose, load_path=load_path, save_path=save_path)
        
        if from_file:
            try:
                self = super().load_from_path()
            except FileNotFoundError:
                raise FileNotFoundError(f'File not found: {self.load_path}')
        
        else:
            super().__init__(steps, memory=memory, verbose=verbose)
            self.steps = steps

            self.device = device

            if self.steps and len(self.steps) > 0:
                self.splitter = self.get_splitter()
                self.pipeline_transformers = self.get_pipeline_transformers()
                self.classifier = self.get_classifier()

    def get_splitter(self):
        for step_name, step in self.steps:
            if issubclass(type(step), BaseEstimator) and type(step) is SplitterPreprocessor:
                return step

        return None
    
    def get_pipeline_transformers(self):
        transformers = []

        for step_name, step in self.steps:
            if issubclass(type(step), TransformerMixin) and type(step) is not SplitterPreprocessor:
                transformers.append((step_name, step))

        transformers = Pipeline(steps=transformers)

        return transformers

    def get_classifier(self):
        for step_name, step in self.steps:
            if issubclass(type(step), ClassifierMixin):
                return (step_name, step)

        return None

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out(input_features)

    def fit(self, df: pd.DataFrame, y: pd.Series = None):
        df_prep = df.copy()
        
        if self.splitter:
            df_prep_train = self.splitter.fit_transform(df_prep)
        else:
            df_prep_train = df_prep

        df_prep_train = self.pipeline_transformers.fit_transform(df_prep_train)

        features_columns = list(self.pipeline_transformers.named_steps['vectorizer'].get_feature_names())

        self.classifier.input_column_names = features_columns

        if self.classifier:
            self.classifier.fit(df_prep_train)

        return df_prep_train

    def transform(self, df: pd.DataFrame, dataset_type: str = 'test'):
        df_prep = df.copy()
        
        if self.splitter:
            self.splitter.fit_transform(df_prep)
            
            df_prep = self.splitter.get_split(dataset_type=dataset_type)
        
        df_prep = self.pipeline_transformers.transform(df_prep)
        
        return df_prep

    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None):
        raise NotImplementedError('PipelineModules does not support fit_transform method')

    def predict(self, df: pd.DataFrame):
        df_prep = df.copy()

        df_prep['prediction'] = self.classifier.predict(df_prep)

        return df_prep

    def predict_proba(self, df: pd.DataFrame):
        df_prep = df.copy()

        # Get probabilities for each class
        probabilities = self.classifier.predict_proba(df_prep)

        # Assuming probabilities is a 2D array, get the probabilities of the predicted class
        predicted_class_probs = probabilities[np.arange(probabilities.shape[0]), self.classifier.predict(df_prep)]

        # Create a new column for the probabilities of the predicted class
        df_prep['prediction_proba'] = predicted_class_probs

        return df_prep

    def evaluate(self, df: pd.DataFrame):
        df_prep = df.copy()

        evaluator = MetricsEvaluator()

        metrics = evaluator.fit_transform(df_prep)

        return metrics
    
    def is_fitted(self):
        validator = False
        if self.splitter:
            validator = validator and self.splitter.is_fitted()
        
        if self.pipeline_transformers:
            validator = validator and self.pipeline_transformers.is_fitted()
        
        if self.classifier:
            validator = validator and self.classifier.is_fitted()
        
        return validator