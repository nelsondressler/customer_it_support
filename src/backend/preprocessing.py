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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

import nltk
import spacy

class EmailPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        prep_steps = [
            self.filter_language,
            self.fill_missing_values,
            self.attach_subject_body,
            self.extract_features
        ]

        for step in prep_steps:
            df_prep = step(df_prep)

        return df_prep

    def filter_language(self, df: pd.DataFrame, lang: str = 'en') -> pd.DataFrame:
        if 'language' not in df.columns:
            df['language'] = df['text'].apply(lambda x: self.detect_language(x))

        df_filtered = df[df['language'] == lang].copy()

        return df_filtered

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.copy()
        df_filled['subject'] = df_filled['subject'].fillna('no subject')
        df_filled['body'] = df_filled['body'].fillna('no body')

        return df_filled

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_featured = df.copy()

        df_featured['subject_length'] = df_featured['subject'].apply(lambda x: len(x))
        df_featured['body_length'] = df_featured['body'].apply(lambda x: len(x))

        return df_featured

    def attach_subject_body(self, df: pd.DataFrame) -> pd.DataFrame:
        df_attached = df.copy()
        df_attached['text'] = df_attached.apply(
            lambda row: f"Subject: {row['subject']}\nBody: {row['body']}",
            axis=1
        )

        return df_attached

    def detect_language(self, text: str, lang: str = 'en') -> str:
        try:
            return langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            return 'unknown'

class ResamplingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        label_columns: str | List[str],
        resample_mode: str = 'undersample',
        random_state: int = 42,
        n_samples_each_category: int = None
    ) -> None:
        if type(label_columns) == str:
            self.label_columns = [label_columns]
        else:
            self.label_columns = label_columns

        self.random_state = random_state
        self.resample_mode = resample_mode
        self.n_samples_each_category = n_samples_each_category

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        prep_steps = [
            self.resample
        ]

        for step in prep_steps:
            df_prep = step(df_prep)

        return df_prep

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        if self.resample_mode == 'undersample':
            df_prep = self.undersample_examples(df=df_prep)
        elif self.resample_mode == 'oversample':
            df_prep = self.oversample_examples(df=df_prep)

        return df_prep

    def undersample_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Undersamples a DataFrame based on a subset of columns to create a balanced dataset.

        Args:
            columns (List[str]): The columns to group the DataFrame by for undersampling.
            n_samples_each_category (int, optional): The number of samples to keep from each category. If None, all samples from each category will be kept. Defaults to None.
            random_state (int, optional): The random state for reproducibility. Defaults to 42.
        """
        df_prep = df.copy()

        if self.n_samples_each_category is None:
            n_samples = df_prep.groupby(self.label_columns).size().min()
        else:
            n_samples = self.n_samples_each_category

        undersampled_dfs = df_prep.groupby(self.label_columns).sample(n=n_samples, replace=False, random_state=self.random_state)

        return undersampled_dfs

    def oversample_examples(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Oversamples a DataFrame based on a subset of columns to create a balanced dataset.

        Args:
            columns (List[str]): The columns to group the DataFrame by for oversampling.
            random_state (int, optional): The random state for reproducibility. Defaults to 42.
        """
        df_prep = df.copy()

        max_samples = df_prep.groupby(self.label_columns).size().max()
        oversampled_dfs = []

        for _, group_df in df_prep.groupby(self.label_columns):
            n_samples = max_samples - len(group_df)

            oversampled_dfs.append(group_df.sample(n=n_samples, replace=True, random_state=self.random_state))

        oversampled_dfs = pd.concat(oversampled_dfs)

        return oversampled_dfs

class SplitterPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        split_mode: str = 'train_val_test',
        retrieve: str = 'all',
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        self.split_mode = split_mode
        self.retrieve = retrieve

        self.test_size = test_size
        self.val_size = val_size

        self.random_state = random_state

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_prep = df.copy()

        if self.split_mode == 'train_val_test':
            df_train, df_val, df_test = self.split_train_val_test(df_prep)
        elif self.split_mode == 'train_test':
            df_train, df_test = self.split_train_test(df_prep)
        elif self.split_mode == 'train_val':
            df_train, df_val = self.split_train_val(df_prep)
        else:
            raise ValueError(f'Invalid value for split_mode: {self.split_mode}')

        if self.retrieve == 'all':
            return df_train, df_val, df_test
        elif self.retrieve == 'train':
            return df_train
        elif self.retrieve == 'val':
            return df_val
        elif self.retrieve == 'test':
            return df_test
        else:
            raise ValueError(f'Invalid value for retrieve: {self.retrieve}')

    def split_train_val_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train_val, df_test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        df_train, df_val = train_test_split(df_train_val, test_size=self.val_size, random_state=self.random_state)

        return df_train, df_val, df_test

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train, df_test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)

        return df_train, df_test

    def split_train_val(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_train, df_val = train_test_split(df, test_size=self.val_size, random_state=self.random_state)

        return df_train, df_val

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        stopwords: List[str] = None,
        flg_stemm: bool = False,
        flg_lemm: bool = False,
        flg_stopwords: bool = True,
        flg_punctuation: bool = True,
        flg_numbers: bool = True
    ) -> None:
        self.flg_stemm = flg_stemm
        self.flg_lemm = flg_lemm
        self.flg_stopwords = flg_stopwords
        self.flg_punctuation = flg_punctuation
        self.flg_numbers = flg_numbers

        self.load_stopwords(stopwords=stopwords)

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()
        df_prep['text'] = df_prep['text'].apply(lambda x: self.preprocess_text(x))

        return df_prep

    def load_stopwords(
        self,
        stopwords: List[str] = None,
        lang: str = 'english',
        source: str = 'nltk',
        file_path: str = ''
    ):
        if stopwords is not None:
            self.stopwords = stopwords
        elif source == 'nltk':
            nltk.download('stopwords')
            nltk.download('wordnet')

            self.stopwords = nltk.corpus.stopwords.words(lang)
        elif source == 'spacy':
            spacy.load('en_core_web_sm')

            self.stopwords = spacy.load(lang).Defaults.stop_words
        elif source == 'file':
            try:
                with open(file_path, 'r') as f:
                    self.stopwords = f.read().splitlines()
            except FileNotFoundError:
                print(f'File not found: {file_path}')
                self.stopwords = []
        else:
            self.stopwords = []

    def preprocess_text(self, text: str) -> str:
        import unicodedata

        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        text = text.lower()

        if self.flg_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        if self.flg_numbers:
            text = re.sub(r'\d+', '', text)

        text = re.sub(r'\s{2,}', ' ', text)

        if self.flg_stemm:
            ps = nltk.stem.porter.PorterStemmer()
            text = ' '.join([ps.stem(word) for word in text.split()])

        if self.flg_lemm:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            text = ' '.join([lem.lemmatize(word) for word in text.split()])

        if self.flg_stopwords:
            stopwords = nltk.corpus.stopwords.words('english')
            text = ' '.join([word for word in text.split() if word not in stopwords])

        text = text.strip()

        return text

class LabelPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, label_column_name: str, encoder_mode: str = 'label') -> None:
        self.label_column_name = label_column_name
        self.encoder_mode = encoder_mode

        if self.encoder_mode == 'label':
            self.encoder = LabelEncoder()
        elif self.encoder_mode == 'onehot':
            self.encoder = OneHotEncoder()

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        if self.label_column_name in df_prep.columns:
            prep_steps = [
                self.set_label,
                self.encode
            ]

            for step in prep_steps:
                df_prep = step(df_prep)

        return df_prep

    def set_label(self, df: pd.DataFrame):
        df_prep = df.copy()
        df_prep['label'] = df_prep[self.label_column_name]

        return df_prep

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        df_prep['label'] = self.encoder.fit_transform(df_prep['label'])

        return df_prep

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        df_prep['label'] = self.encoder.inverse_transform(df_prep['label'])

        return df_prep

class VectorizerPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        from_file: bool = False,
        file_path: str = '',
        vectorizer_mode: str = 'TfIdfVectorizer'
    ) -> None:
        self.from_file = from_file
        self.file_path = file_path
        self.vectorizer_mode = vectorizer_mode

        if from_file:
            try:
                self.vectorizer = self.load_vectorizer(file_path)
                self.vectorizer_mode = self.vectorizer.__class__.__name__
            except FileNotFoundError:
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), smooth_idf=True, use_idf=True)
                self.vectorizer_mode = self.vectorizer.__class__.__name__

        elif self.vectorizer_mode == 'TfIdfVectorizer':
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), smooth_idf=True, use_idf=True)
        elif self.vectorizer_mode == 'CountVectorizer':
            self.vectorizer = CountVectorizer()
        else:
            raise ValueError(f'Invalid value for vectorizer_mode: {self.vectorizer_mode}')

    def fit(self, df: pd.DataFrame):
        df_prep = df.copy()

        if self.vectorizer:
            self.vectorizer.fit(df_prep['text'])

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        if self.vectorizer:
            # Get the sparse matrix from the vectorizer
            feature_matrix = self.vectorizer.transform(df_prep['text'])
            # Convert sparse matrix to dense array and create a DataFrame
            # feature_df = pd.DataFrame(feature_matrix.toarray())
            feature_df = pd.DataFrame(feature_matrix.toarray(), columns=self.vectorizer.get_feature_names_out())

            for col in df_prep.columns:
                if col in feature_df.columns:
                    df_prep.rename(columns={col: f'{col}_original'}, inplace=True)

            # Concatenate feature DataFrame with original DataFrame
            df_prep = pd.concat(
                [df_prep.reset_index(drop=True), feature_df],
                axis=1
            ) # Concatenate feature columns

        else:
            df_prep['features'] = df_prep['text']

        return df_prep

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def get_vocabulary(self):
        return self.vectorizer.vocabulary_

    def load_vectorizer(self, file_path: str):
        try:
            with open(file_path, 'rb') as f:
                vectorizer = pickle.load(f)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            vectorizer = None

        return vectorizer

    def save_vectorizer(self, file_path: str):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            vectorizer = None