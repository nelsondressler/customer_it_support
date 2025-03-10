import os

import pandas as pd

from fastapi import FastAPI, HTTPException
import uvicorn

import config

from saved_datasets.data_examples import email_examples

from src.backend.evaluation import MetricsEvaluator
from src.backend.preprocessing import SplitterPreprocessor, EmailPreprocessor, ResamplingPreprocessor, TextPreprocessor, LabelPreprocessor, VectorizerPreprocessor
from src.backend.models import BaselineModel, TransformerModel
from src.backend.pipelines import PipelineModules

from src.utils.pydantic_models import EmailInput, PredictionResponse
from src.utils.labels import label_queue_values, label_priority_values
from src.utils.devices import get_available_device

import wandb

wandb.login()

data_df = pd.DataFrame(email_examples)

splitter = SplitterPreprocessor(retrieve='all')

data_prep_train_df, data_prep_val_df, data_prep_text_df = splitter.fit_transform(data_df)

app = FastAPI(title="Customer IT Support Prediction API")

@app.get("/")
def send_welcome():
    return "Welcome to Customer IT Support Prediction API!"

@app.post("/predict", response_model=PredictionResponse)
def predict(email: EmailInput):
    LOAD_MODE = os.getenv('LOAD_MODE')
    FIT_FLG = eval(os.getenv('FIT_FLG'))

    # Preprocess the input text
    dict_email = {
        'language': 'en',
        'subject': email.subject,
        'body': email.body
    }
    df_email = pd.DataFrame(dict_email, index=[0])

    if email.model_choice.lower() in ["nb", "lr"]:
        if LOAD_MODE == 'model':
            vectorizer_path = os.getenv('VECTORIZER_PATH')
            vectorizer = VectorizerPreprocessor(from_file=True, file_path=vectorizer_path)

        if email.model_choice.lower() == 'nb':
            if LOAD_MODE == 'pipeline':
                nb_pipeline_queue_path = os.getenv('NB_PIPELINE_QUEUE_PATH')
                nb_pipeline_priority_path = os.getenv('NB_PIPELINE_PRIORITY_PATH')

                pipeline_queue = PipelineModules(
                    from_file=True,
                    file_path=nb_pipeline_queue_path
                )

                pipeline_priority = PipelineModules(
                    from_file=True,
                    file_path=nb_pipeline_priority_path
                )
            elif LOAD_MODE == 'model':
                nb_model_queue_path = os.getenv('NB_MODEL_QUEUE_PATH')
                nb_model_priority_path = os.getenv('NB_MODEL_PRIORITY_PATH')

                model_queue = BaselineModel(from_file=True, file_path=nb_model_queue_path)
                model_priority = BaselineModel(from_file=True, file_path=nb_model_priority_path)

                pipeline_queue = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('vectorizer', vectorizer),
                    ('classifier', model_queue)
                ])

                pipeline_priority = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('vectorizer', vectorizer),
                    ('classifier', model_priority)
                ])

        elif email.model_choice.lower() == 'lr':
            if LOAD_MODE == 'pipeline':
                lr_pipeline_queue_path = os.getenv('LR_PIPELINE_QUEUE_PATH')
                lr_pipeline_priority_path = os.getenv('LR_PIPELINE_PRIORITY_PATH')

                pipeline_queue = PipelineModules(
                    from_file=True,
                    file_path=lr_pipeline_queue_path
                )

                pipeline_priority = PipelineModules(
                    from_file=True,
                    file_path=lr_pipeline_priority_path
                )
            elif LOAD_MODE == 'model':
                lr_model_queue_path = os.getenv('LR_MODEL_QUEUE_PATH')
                lr_model_priority_path = os.getenv('LR_MODEL_PRIORITY_PATH')

                model_queue = BaselineModel(model_path=lr_model_queue_path)
                model_priority = BaselineModel(model_path=lr_model_priority_path)

                pipeline_queue = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('vectorizer', vectorizer),
                    ('classifier', model_queue)
                ])

                pipeline_priority = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('vectorizer', vectorizer),
                    ('classifier', model_priority)
                ])

        if pipeline_queue.steps is None or pipeline_priority.steps is None:
            raise HTTPException(status_code=400, detail="Pipeline not found")

        if FIT_FLG:
            pipeline_queue = PipelineModules(steps=[
                ('email_preprocessor', EmailPreprocessor()),
                ('text_preprocessor', TextPreprocessor()),
                ('label_preprocessor', LabelPreprocessor(label_column_name='queue')),
                ('vectorizer', VectorizerPreprocessor()),
                ('classifier', BaselineModel(model='LogisticRegression')) if email.model_choice.lower() == 'lr' else BaselineModel(model='MultinomialNB')
            ])
            pipeline_queue.fit(data_prep_train_df)

            pipeline_priority = PipelineModules(steps=[
                ('email_preprocessor', EmailPreprocessor()),
                ('text_preprocessor', TextPreprocessor()),
                ('label_preprocessor', LabelPreprocessor(label_column_name='priority')),
                ('vectorizer', VectorizerPreprocessor()),
                ('classifier', BaselineModel(model='LogisticRegression')) if email.model_choice.lower() == 'lr' else BaselineModel(model='MultinomialNB')
            ])
            pipeline_priority.fit(data_prep_train_df)

    elif "bert" in email.model_choice.lower():
        device = get_available_device()

        if email.model_choice.lower() == 'bert':
            if LOAD_MODE == 'pipeline':
                bert_pipeline_queue_path = os.getenv('BERT_PIPELINE_QUEUE_PATH')
                bert_pipeline_priority_path = os.getenv('BERT_PIPELINE_PRIORITY_PATH')

                pipeline_queue = PipelineModules(
                    from_file=True,
                    file_path=bert_pipeline_queue_path,
                    device=device
                )

                pipeline_priority = PipelineModules(
                    from_file=True,
                    file_path=bert_pipeline_priority_path,
                    device=device
                )
            elif LOAD_MODE == 'model':
                bert_model_queue_path = os.getenv('BERT_TRANSFORMERS_MODEL_QUEUE_PATH')
                bert_model_priority_path = os.getenv('BERT_TRANSFORMERS_MODEL_PRIORITY_PATH')

                model_queue = TransformerModel(from_filte=True, load_path=bert_model_queue_path, device=device)
                model_priority = TransformerModel(from_file=True, load_path=bert_model_priority_path, device=device)

                pipeline_queue = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('classifier', model_queue)
                ])

                pipeline_priority = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('classifier', model_priority)
                ])

        elif email.model_choice.lower() == 'distilbert':
            if LOAD_MODE == 'pipeline':
                distilbert_pipeline_queue_path = os.getenv('DISTILBERT_PIPELINE_QUEUE_PATH')
                distilbert_pipeline_priority_path = os.getenv('DISTILBERT_PIPELINE_PRIORITY_PATH')

                pipeline_queue = PipelineModules(
                    from_file=True,
                    file_path=distilbert_pipeline_queue_path,
                    device=device
                )

                pipeline_priority = PipelineModules(
                    from_file=True,
                    file_path=distilbert_pipeline_priority_path,
                    device=device
                )

            elif LOAD_MODE == 'model':
                distilbert_model_queue_path = os.getenv('DISTILBERT_TRANSFORMERS_MODEL_QUEUE_PATH')
                distilbert_model_priority_path = os.getenv('DISTILBERT_TRANSFORMERS_MODEL_PRIORITY_PATH')

                model_queue = TransformerModel(model_path=distilbert_model_queue_path, device=device)
                model_priority = TransformerModel(model_path=distilbert_model_priority_path, device=device)

                pipeline_queue = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('classifier', model_queue)
                ])

                pipeline_priority = PipelineModules(steps=[
                    ('email_preprocessor', EmailPreprocessor()),
                    ('text_preprocessor', TextPreprocessor()),
                    ('classifier', model_priority)
                ])

        if FIT_FLG:
            pipeline_queue = PipelineModules(steps=[
                ('email_preprocessor', EmailPreprocessor()),
                ('text_preprocessor', TextPreprocessor()),
                ('label_preprocessor', LabelPreprocessor(label_column_name='queue')),
                ('classifier', TransformerModel(model='bert-base-uncased', device=device)) if email.model_choice.lower() == 'bert' else TransformerModel(model='distilbert-base-uncased', device=device)
            ])
            pipeline_queue.fit(data_prep_train_df)

            pipeline_priority = PipelineModules(steps=[
                ('email_preprocessor', EmailPreprocessor()),
                ('text_preprocessor', TextPreprocessor()),
                ('label_preprocessor', LabelPreprocessor(label_column_name='priority')),
                ('classifier', TransformerModel(model='bert-base-uncased', device=device)) if email.model_choice.lower() == 'bert' else TransformerModel(model='distilbert-base-uncased', device=device)
            ])
            pipeline_priority.fit(data_prep_train_df)

    else:
        raise HTTPException(status_code=400, detail="Invalid model choice")

    df_prep_email = df_email
    df_prep_email = pipeline_queue.transform(df_prep_email)
    df_prep_email = pipeline_queue.predict(df_prep_email)
    pred_queue = df_prep_email['prediction'].values[0]
    df_prep_email = pipeline_queue.predict_proba(df_prep_email)
    pred_queue_proba = df_prep_email['prediction_proba'].values[0]

    df_prep_email = df_email
    df_prep_email = pipeline_priority.transform(df_prep_email)
    df_prep_email = pipeline_priority.predict(df_prep_email)
    pred_priority = df_prep_email['prediction'].values[0]
    df_prep_email = pipeline_priority.predict_proba(df_prep_email)
    pred_priority_proba = df_prep_email['prediction_proba'].values[0]

    details = {
        "model": email.model_choice,
        "dataframe_shape": df_prep_email.shape,
        "predict_proba_queue": pred_queue_proba.tolist(),
        "predict_proba_priority": pred_priority_proba.tolist()
    }

    pred_queue_name = label_queue_values[pred_queue]
    pred_priority_name = label_priority_values[pred_priority]

    return PredictionResponse(queue=pred_queue_name, priority=pred_priority_name, details=details)