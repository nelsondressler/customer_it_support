import os
import sys
import pickle

from typing import List, Tuple, Dict, Any, Optional, Union

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve

from src.backend.evaluation import MetricsEvaluator

import nltk
import spacy

class PipelineModules(Pipeline):
    def __init__(
        self,
        from_file: bool = False,
        file_path: str = '',
        steps: Tuple[str, Any] = None,
        memory: str = None,
        device: str = 'cpu',
        verbose: bool = False
    ) -> None:
        if from_file:
            try:
                self.steps = self.load_pipeline(file_path)
            except FileNotFoundError:
                super().__init__(steps, memory=memory, verbose=verbose)
                self.steps = steps
        else:
            super().__init__(steps, memory=memory, verbose=verbose)
            self.steps = steps

        self.device = device

        if self.steps and len(self.steps) > 0:
            print(self.steps)
            self.pipeline_transformers = self.get_pipeline_transformers()
            self.classifier = self.get_classifier()

    def get_pipeline_transformers(self):
        transformers = []

        for step_name, step in self.steps:
            if issubclass(type(step), TransformerMixin):
                transformers.append((step_name, step))

        transformers = Pipeline(transformers)

        return transformers

    def get_classifier(self):
        classifiers = []

        for step_name, step in self.steps:
            if issubclass(type(step), ClassifierMixin):
                classifiers.append((step_name, step))
                # if isinstance(type(step), TransformerModel):
                #     step.device = self.device
                #     model = TransformerModel(**step.kwargs)
                #     processed_step = processed_step.load_models()
                #     classifiers.append((step_name, processed_step))
                # else:
                #     classifiers.append((step_name, step))

        if len(classifiers) == 0:
            raise ValueError('No classifier found in the pipeline')
        else:
            classifier = classifiers[0][1]

        return classifier

    def get_feature_names_out(self, input_features=None):
        return super().get_feature_names_out(input_features)

    def fit(self, df: pd.DataFrame, y: pd.Series = None):
        df_prep = df.copy()

        df_prep = self.pipeline_transformers.fit_transform(df_prep)

        features_columns = list(self.pipeline_transformers.named_steps['vectorizer'].get_feature_names())

        self.classifier.input_column_names = features_columns

        self.classifier.fit(df_prep)

        return df_prep

    def transform(self, df: pd.DataFrame):
        df_prep = df.copy()

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

    def load_pipeline(self, file_path: str):
        try:
            with open(file_path, 'rb') as f:
                pipeline = pickle.load(f)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            pipeline = None

        return pipeline

    def save_pipeline(self, file_path: str):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            pipeline = None