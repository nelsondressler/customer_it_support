from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from transformers import BertModel, DistilBertModel

from backend.preprocessors import (
    SplitterPreprocessor,
    EmailPreprocessor,
    ResamplingPreprocessor,
    TextPreprocessor,
    LabelPreprocessor,
    VectorizerPreprocessor
)
# from src.backend.models import BaselineModel, TransformerModel
from src.backend.pipelines import PipelineModules

# Artifacts
artifacts_types = {
    'SplitterPreprocessor': 'Preprocessor',
    'EmailPreprocessor': 'Preprocessor',
    'ResamplingPreprocessor': 'Preprocessor',
    'TextPreprocessor': 'Preprocessor',
    'LabelPreprocessor': 'Preprocessor',
    'VectorizerPreprocessor': 'Preprocessor',
    'BaselineModel': 'Model',
    'TransformerModel': 'Model',
    'PipelineModules': 'Pipeline'
}

# Preprocessors
preprocessor_mapping = {
    'SplitterPreprocessor': SplitterPreprocessor,
    'EmailPreprocessor': EmailPreprocessor,
    'ResamplingPreprocessor': ResamplingPreprocessor,
    'TextPreprocessor': TextPreprocessor,
    'LabelPreprocessor': LabelPreprocessor,
    'VectorizerPreprocessor': VectorizerPreprocessor
}

preprocessor_types = {
    'SplitterPreprocessor': 'Splitter',
    'EmailPreprocessor': 'EmailBased',
    'ResamplingPreprocessor': 'Resampler',
    'TextPreprocessor': 'TextBased',
    'LabelPreprocessor': 'LabelBased',
    'VectorizerPreprocessor': 'Vectorizer'
}

preprocessor_abbreviations = {
    'SplitterPreprocessor': 'sp',
    'EmailPreprocessor': 'ep',
    'ResamplingPreprocessor': 'rp',
    'TextPreprocessor': 'tp',
    'LabelPreprocessor': 'lp',
    'VectorizerPreprocessor': 'vp'
}

# Models
models_mapping = {
    'MultinomialNB': MultinomialNB,
    'BernoulliNB': BernoulliNB,
    'GaussianNB': GaussianNB,
    'LogisticRegression': LogisticRegression,
    'KNeighborsClassifier': KNeighborsClassifier,
    'SVC': SVC,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'bert-base-uncased': BertModel,
    'distilbert-base-uncased': DistilBertModel
}

models_types = {
    'MultinomialNB': 'Baseline',
    'BernoulliNB': 'Baseline',
    'GaussianNB': 'Baseline',
    'LogisticRegression': 'Baseline',
    'KNeighborsClassifier': 'Baseline',
    'SVC': 'Baseline',
    'DecisionTreeClassifier': 'Baseline',
    'RandomForestClassifier': 'Baseline',
    'GradientBoostingClassifier': 'Baseline',
    'bert-base-uncased': 'Transformer',
    'distilbert-base-uncased': 'Transformer'
}

models_abbreviations = {
    'MultinomialNB': 'nb',
    'BernoulliNB': 'bn',
    'GaussianNB': 'gn',
    'LogisticRegression': 'lr',
    'KNeighborsClassifier': 'knc',
    'SVC': 'svc',
    'DecisionTreeClassifier': 'dtc',
    'RandomForestClassifier': 'rfc',
    'GradientBoostingClassifier': 'gbc'
}

#Pipelines
pipelines_mapping = {
    'BaselineTrainDatasetPipeline': PipelineModules(steps=[
        ('SplitterPreprocessorPipeline', SplitterPreprocessor(retrieve='train')),
        ('EmailPreprocessor', EmailPreprocessor()),
        ('ResamplingPreprocessor', ResamplingPreprocessor()),
        ('TextPreprocessor', TextPreprocessor()),
        ('VectorizerPreprocessor', VectorizerPreprocessor()),
        ('LabelPreprocessor', LabelPreprocessor())
    ]),
    'BaselineTestDatasetPipeline': PipelineModules(steps=[
        ('SplitterPreprocessor', SplitterPreprocessor(retrieve='test')),
        ('EmailPreprocessor', EmailPreprocessor()),
        ('ResamplingPreprocessor', ResamplingPreprocessor()),
        ('TextPreprocessor', TextPreprocessor()),
        ('VectorizerPreprocessor', VectorizerPreprocessor()),
        ('LabelPreprocessor', LabelPreprocessor())
    ]),
    'TransformerTrainDatasetPipeline': PipelineModules(steps=[
        ('SplitterPreprocessor', SplitterPreprocessor(retrieve='train')),
        ('EmailPreprocessor', EmailPreprocessor()),
        ('ResamplingPreprocessor', ResamplingPreprocessor()),
        ('TextPreprocessor', TextPreprocessor()),
        ('LabelPreprocessor', LabelPreprocessor())
    ]),
    'TransformerTestDatasetPipeline': PipelineModules(steps=[
        ('SplitterPreprocessor', SplitterPreprocessor(retrieve='test')),
        ('EmailPreprocessor', EmailPreprocessor()),
        ('ResamplingPreprocessor', ResamplingPreprocessor()),
        ('TextPreprocessor', TextPreprocessor()),
        ('LabelPreprocessor', LabelPreprocessor())
    ])
}

pipelines_types = {
    'TraditionalMLTrainDatasetPipeline': 'TraditionalMLPipeline',
    'TraditionalMLTestDatasetPipeline': 'TraditionalMLPipeline',
    'TransformerTrainDatasetPipeline': 'TransformerPipeline',
    'TransformerTestDatasetPipeline': 'TransformerPipeline'
}

pipelines_abbreviations = {
    'TraditionalMLTrainDatasetPipeline': 'tmt',
    'TraditionalMLTestDatasetPipeline': 'tmte',
    'TransformerTrainDatasetPipeline': 'tt',
    'TransformerTestDatasetPipeline': 'tte'
}

artifacts_mapping = {
    'Preprocessor': preprocessor_mapping,
    'Model': models_mapping,
    'Pipeline': pipelines_mapping
}

artifacts_abbreviations = {
    'Preprocessor': preprocessor_abbreviations,
    'Model': models_abbreviations,
    'Pipeline': pipelines_abbreviations
}

def get_artifact_type(artifact_name: str):
    return artifacts_types[artifact_name]

def get_artifact_estimator(artifact_name: str, artifact_type: str):
    return artifacts_mapping[artifact_type][artifact_name]

def get_artifact_abbreviation(artifact_name: str, artifact_type: str):
    return artifacts_abbreviations[artifact_type][artifact_name]

def artifact_is_model(artifact_name: str):
    return artifacts_types[artifact_name] == 'Model'

def artifact_is_transformer(artifact_name: str):
    return models_types[artifact_name] == 'Transformer'

def artifact_is_pipeline(artifact_name: str):
    return artifacts_types[artifact_name] == 'Pipeline'