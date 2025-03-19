import os
from typing import Union

import pandas as pd
from sklearn.base import BaseEstimator

from backend.artifacts import Artifact

import config


class EmailController:
    def __init__(
        self,
        from_file: bool = False,
        artifact_name: str = 'LogisticRegression',
        output_column_name: str = 'queue',
        **kwargs
    ):
        self.artifact = Artifact(
            from_file=from_file,
            folder_path=config.ARTIFACTS_FOLDER,
            name=artifact_name,
            label=output_column_name,
            **kwargs
        )
        
        if not self.artifact.is_fitted():
            self.fit()
    
    def get_dataset(self):
        if os.path.exists(config.HF_DATASETS_CACHE_PATH):
            df = pd.read_csv(config.HF_DATASETS_CACHE_PATH)
        elif not os.path.exists(config.HF_DATASETS_CACHE_PATH):
            df = pd.read_csv(config.HF_DATASETS_PATH)
            
        return df
    
    def fit(self, df: pd.DataFrame = None):
        if df is not None:
            self.df = df
        else:
            self.df = self.get_dataset()
        
        self.artifact.fit(df=self.df)
        self.artifact.save_to_path()
    
    def predict(self, subject: str, body: str) -> Union[pd.DataFrame, BaseEstimator]:
        dict_email = {
            'language': 'en',
            'subject': subject,
            'body': body
        }
        df_email = pd.DataFrame(dict_email, index=[0])
        
        return self.artifact.predict(df=df_email)['predictions'][0]