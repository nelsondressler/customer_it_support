import os
import sys

from dotenv import load_dotenv

from utils.paths import add_prefix_path

load_dotenv()

API_URL = os.getenv('API_URL')
UI_GRADIO_URL = os.getenv('UI_GRADIO_URL')
UI_STREAMLIT_URL = os.getenv('UI_STREAMLIT_URL')

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    PREFIX_PATH = '/content/drive/MyDrive/customer-support-tickets/'
else:
    PREFIX_PATH = ''

LOAD_MODE = os.getenv('LOAD_MODE')
FIT_FLG = eval(os.getenv('FIT_FLG'))

VECTORIZER_PATH = add_prefix_path(file_path=os.getenv('VECTORIZER_PATH'), prefix_path=PREFIX_PATH)

NB_PIPELINE_QUEUE_PATH = add_prefix_path(file_path=os.getenv('NB_PIPELINE_QUEUE_PATH'), prefix_path=PREFIX_PATH)
NB_PIPELINE_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('NB_PIPELINE_PRIORITY_PATH'), prefix_path=PREFIX_PATH)

NB_MODEL_QUEUE_PATH = add_prefix_path(file_path=os.getenv('NB_MODEL_QUEUE_PATH'), prefix_path=PREFIX_PATH)
NB_MODEL_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('NB_MODEL_PRIORITY_PATH'), prefix_path=PREFIX_PATH)

LR_PIPELINE_QUEUE_PATH = add_prefix_path(file_path=os.getenv('LR_PIPELINE_QUEUE_PATH'), prefix_path=PREFIX_PATH)
LR_PIPELINE_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('LR_PIPELINE_PRIORITY_PATH'), prefix_path=PREFIX_PATH)

LR_MODEL_QUEUE_PATH = add_prefix_path(file_path=os.getenv('LR_MODEL_QUEUE_PATH'), prefix_path=PREFIX_PATH)
LR_MODEL_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('LR_MODEL_PRIORITY_PATH'), prefix_path=PREFIX_PATH)

BERT_PIPELINE_QUEUE_PATH = add_prefix_path(file_path=os.getenv('BERT_PIPELINE_QUEUE_PATH'), prefix_path=PREFIX_PATH)
BERT_PIPELINE_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('BERT_PIPELINE_PRIORITY_PATH'), prefix_path=PREFIX_PATH)

BERT_MODEL_QUEUE_PATH = add_prefix_path(file_path=os.getenv('BERT_MODEL_QUEUE_PATH'), prefix_path=PREFIX_PATH)
BERT_MODEL_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('BERT_MODEL_PRIORITY_PATH'), prefix_path=PREFIX_PATH)

DISTILBERT_PIPELINE_QUEUE_PATH = add_prefix_path(file_path=os.getenv('DISTILBERT_PIPELINE_QUEUE_PATH'), prefix_path=PREFIX_PATH)
DISTILBERT_PIPELINE_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('DISTILBERT_PIPELINE_PRIORITY_PATH'), prefix_path=PREFIX_PATH)

DISTILBERT_MODEL_QUEUE_PATH = add_prefix_path(file_path=os.getenv('DISTILBERT_MODEL_QUEUE_PATH'), prefix_path=PREFIX_PATH)
DISTILBERT_MODEL_PRIORITY_PATH = add_prefix_path(file_path=os.getenv('DISTILBERT_MODEL_PRIORITY_PATH'), prefix_path=PREFIX_PATH)