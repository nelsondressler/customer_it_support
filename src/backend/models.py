import os
import sys
import pickle

from typing import List, Tuple, Dict, Any, Optional, Union

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve

import nltk
import spacy

from datasets import Dataset

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

import wandb
import config

from src.utils.artifacts import models_mapping

class BaseModel(BaseEstimator, ClassifierMixin):
    def __init__(self, load_path: str, save_path: str) -> None:
        self.load_path = load_path
        self.save_path = save_path
    
    def load_from_path(self):
        try:
            with open(self.load_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found: {self.load_path}')
    
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
    

class BaselineModel(BaseEstimator):
    models_mapping = models_mapping
    
    def __init__(
        self,
        from_file: bool = False,
        load_path: str = '',
        save_path: str = '',
        model: str = 'LogisticRegression',
        input_column_names_numeric: bool = False,
        input_column_names: List[str] | str = 'text',
        output_column_name: str = 'label',
        device: str = 'cpu',
        **kwargs
    ) -> None:
        super().__init__(load_path=load_path, save_path=save_path)
        
        if from_file:
            try:
                self = super().load_from_path()
            except FileNotFoundError:
                raise FileNotFoundError(f'File not found: {self.load_path}')
        
        else:
            self.model = self.models_mapping[model](**kwargs)
            self.model_name = model
        
        self.device = device

        self.kwargs = kwargs

        self.input_column_names_numeric = input_column_names_numeric # Store the argument as an attribute

        if input_column_names_numeric:
            self.input_column_names = None
        elif type(input_column_names) is str:
            self.input_column_names = [input_column_names]
        else:
            self.input_column_names = input_column_names

        self.output_column_name = output_column_name

    def get_numeric_columns(self, df: pd.DataFrame):
        return [col for col in df.columns if str(col).isnumeric()]

    def fit(self, df: pd.DataFrame, y: pd.Series = None):
        df_prep = df.copy()
        
        if self.input_column_names_numeric and self.input_column_names is None:
            self.input_column_names = self.get_numeric_columns(df_prep)

        X = df_prep[self.input_column_names]
        y = y if y is not None else df_prep[self.output_column_name]

        self.model.fit(X, y)

        return self

    def predict(self, df: pd.DataFrame):
        df_prep = df.copy()
        
        if self.input_column_names_numeric and self.input_column_names is None:
            self.input_column_names = self.get_numeric_columns(df_prep)

        X = df_prep[self.input_column_names]

        return self.model.predict(X)

    def score(self, df: pd.DataFrame, y: pd.Series = None):
        df_prep = df.copy()
        
        if self.input_column_names_numeric and self.input_column_names is None:
            self.input_column_names = self.get_numeric_columns(df_prep)

        X = df_prep[self.input_column_names]
        y = y if y is not None else df_prep[self.output_column_name]

        return self.model.score(X, y)

    def predict_proba(self, df: pd.DataFrame):
        df_prep = df.copy()
        
        if self.input_column_names_numeric and self.input_column_names is None:
            self.input_column_names = self.get_numeric_columns(df_prep)

        X = df_prep[self.input_column_names]

        return self.model.predict_proba(X)

    def get_params(self, deep: bool = True):
        return self.model.get_params(deep)

    def set_params(self, **params):
        return self.model.set_params(**params)


class TransformerModel(BaseModel):
    models_mapping = models_mapping
    
    def __init__(
        self,
        load_path: str = 'bert-base-uncased',
        save_path: str = './saved_models',
        run_name: str = 'bert-base-uncased',
        input_column_name: str = 'text',
        output_column_name: str = 'label',
        num_labels: int = 2,
        device: str = 'cpu',
        output_dir: str = './results',
        logging_dir: str = './logs',
        epochs: int=3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        **kwargs
    ) -> None:
        super().__init__(load_path=load_path, save_path=save_path)
        
        self.load_path = load_path
        self.save_path = save_path

        self.num_labels = num_labels
        self.device = device
        
        self.model = self.load_from_path()

        self.run_name = run_name

        self.input_column_name = input_column_name
        self.output_column_name = output_column_name

        self.output_dir = output_dir
        self.logging_dir = logging_dir

        self.epochs = epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        if config.WANDB_DISABLED:
            self.wandb = None
            self.report_to = "all"
        else:
            self.wandb = wandb.init(
                project=config.WANDB_PROJECT,
                name=self.run_name,
                entity=config.WANDB_ENTITY,
                config={
                    'epochs': self.epochs,
                    'per_device_train_batch_size': self.per_device_train_batch_size,
                    'per_device_eval_batch_size': self.per_device_eval_batch_size,
                    'learning_rate': self.learning_rate,
                    'weight_decay': self.weight_decay
                }
            )
            self.report_to = "wandb"

    def load_from_path(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.load_path, use_fast=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.load_path, use_fast=False)

        return AutoModelForSequenceClassification.from_pretrained(self.load_path, num_labels=self.num_labels).to(self.device)

    def save_to_path(self):
        self.trainer.save_model(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)
        self.model.save_pretrained(self.save_path)

    def tokenize(self, texts: Dict[str, list], max_length: int = 128):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length)

    def compute_intermediate_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        metrics = {}

        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        metrics['accuracy'] = accuracy_score(y_true=labels, y_pred=predictions)
        metrics['precision'], metrics['recall'], metrics['f1'], _ = precision_recall_fscore_support(y_true=labels, y_pred=predictions, average="weighted")

        return metrics

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame, y: pd.Series = None):
        train_dataset = Dataset.from_pandas(pd.DataFrame(
            {
                'text': df_train[self.input_column_name],
                'label': df_train[self.output_column_name]
            }
        ))
        val_dataset = Dataset.from_pandas(pd.DataFrame(
            {
                'text': df_val[self.input_column_name],
                'label': df_val[self.output_column_name]
            }
        ))

        tokenized_train_dataset = train_dataset.map(self.tokenize, batched=True)
        tokenized_val_dataset = val_dataset.map(self.tokenize, batched=True)

        training_args = TrainingArguments(
            run_name=self.run_name,

            output_dir=self.output_dir,
            logging_dir=self.logging_dir,

            report_to=self.report_to,

            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,

            eval_strategy="epoch",
            save_strategy="epoch",

            load_best_model_at_end=True
        )

        # Create Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            processing_class=self.tokenizer,
            compute_metrics=self.compute_intermediate_metrics
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_results = trainer.evaluate()

        self.trainer = trainer

        if self.wandb:
            self.wandb.log(eval_results)
            
            # [optional] Finish the wandb run, necessary in notebooks
            self.wandb.finish()
            
        self.save_models()

    def compute_predictions_details(self, df: pd.DataFrame):
        df_prep = df.copy()

        # Tokenize and prepare input
        df_prep['embeddings'] = df_prep[self.input_column_name].apply(
            lambda text: self.tokenizer(
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
        )

        # Ensure model is in evaluation mode
        self.model.eval()

        # Run inference
        with torch.no_grad():
            df_prep['logits'] = df_prep['embeddings'].apply(lambda inputs: self.model(**inputs).logits)
            df_prep['predictions'] = df_prep['logits'].apply(lambda logits: torch.argmax(logits, dim=-1).item())

        return df_prep


    def predict(self, df: pd.DataFrame):
        df_prep = df.copy()

        df_prep = self.compute_predictions_details(df_prep)

        return df_prep['predictions']

    def predict_proba(self, df: pd.DataFrame):
        df_prep = df.copy()

        df_prep = self.compute_predictions_details(df_prep)

        return df_prep['logits']