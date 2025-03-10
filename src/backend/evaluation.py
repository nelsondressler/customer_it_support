import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve

class MetricsEvaluator(BaseEstimator, TransformerMixin):
    def __init__(self, label_column_name: str = 'label') -> None:
        self.label_column_name = label_column_name

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prep = df.copy()

        df_prep = self.calculate_metrics(df_prep)

        return df_prep

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        metrics = {}

        metrics['confusion_matrix'] = confusion_matrix(df[self.label_column_name], df['prediction'])
        metrics['classification_report'] = classification_report(df[self.label_column_name], df['prediction'])

        metrics['accuracy'] = accuracy_score(df[self.label_column_name], df['prediction'])
        metrics['precision'], metrics['recall'], metrics['f1_score'], _ = precision_recall_fscore_support(df[self.label_column_name], df['prediction'], average='weighted')

        return metrics