import os

import config

from utils.artifacts import (
    get_artifact_type,
    get_artifact_estimator,
    get_artifact_abbreviation,
    artifact_is_model,
    artifact_is_transformer,
    artifact_is_pipeline
)

from src.backend import preprocessors, models, pipelines

class Artifact:
    def __init__(
        self,
        from_file: bool = False,
        folder_path: str = None,
        name: str = 'LogisticRegression',
        label: str = 'label',
        **kwargs
    ):
        
        self.from_file = from_file
        self.folder_path = folder_path
        self.name = name
        self.label = label
        
        self.type = self.get_type()
        self.abbreviation = self.get_abbreviation()
        self.filename = self.get_filename()
        self.file_path = self.get_path() 
        
        if from_file:
            self.estimator = self.get_estimator()
        else:
            self.estimator = self.get_artifact_default()(**kwargs)
        
    def get_type(self):
        return get_artifact_type(artifact_name=self.name)

    def get_abbreviation(self):
        return get_artifact_abbreviation(artifact_name=self.name, artifact_type=self.type)
    
    def get_filename(self):
        if artifact_is_model(artifact_name=self.name) and artifact_is_transformer(artifact_name=self.name):
            return f'{self.abbreviation}_{self.label}'
        else:
            return f'{self.abbreviation}_{self.label}.pkl'
    
    def get_path(self):
        return os.path.join(self.folder_path, self.filename)
    
    def get_estimator(self):
        if self.from_file:
            try:
                return self.load_from_path()
            except FileNotFoundError:
                raise FileNotFoundError(f'File not found: {self.file_path}')
        else:
            return self.get_artifact_default()

    def load_from_path(self):
        if self.type == 'Preprocessor':
            artifact = preprocessors.load_from_path(file_path=self.file_path)
        elif self.type == 'Model':
            artifact = models.load_from_path(file_path=self.file_path)
        elif self.type == 'Pipeline':
            artifact = pipelines.load_from_path(file_path=self.file_path)

        return artifact
    
    def save_to_path(self):
        if self.type == 'Preprocessor':
            preprocessors.save_to_path(file_path=self.file_path)
        elif self.type == 'Model':
            models.save_to_path(file_path=self.file_path)
        elif self.type == 'Pipeline':
            pipelines.save_to_path(file_path=self.file_path)

    def get_artifact_default(self):
        return get_artifact_estimator(artifact_name=self.name, artifact_type=self.type)
    
    def is_fitted(self):
        return self.estimator.is_fitted()