import os
import sys

from dotenv import load_dotenv

from utils.paths import add_prefix_path

load_dotenv()

WANDB_DISABLED = eval(os.getenv('WANDB_DISABLED'))

if not WANDB_DISABLED:
    import wandb
    
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
    WANDB_PROJECT = os.getenv('WANDB_PROJECT')
    WANDB_LOG_MODEL = os.getenv('WANDB_LOG_MODEL')
    WANDB_WATCH = eval(os.getenv('WANDB_WATCH'))

    wandb.login()

HF_TOKEN = os.getenv('HF_TOKEN')


API_URL = os.getenv('API_URL')
UI_GRADIO_URL = os.getenv('UI_GRADIO_URL')
UI_STREAMLIT_URL = os.getenv('UI_STREAMLIT_URL')

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    PREFIX_PATH = '/content/drive/MyDrive/customer_it_support'
else:
    PREFIX_PATH = '/Users/nelsondressler/Documents/Matrix DNA/Technical Tests/MLEngineer/customer_it_support'

LOAD_MODE = os.getenv('LOAD_MODE')
FIT_FLG = eval(os.getenv('FIT_FLG'))

HF_DATASETS_PATH = add_prefix_path(file_path=os.getenv('HF_DATASETS_PATH'), prefix_path=PREFIX_PATH)
HF_DATASETS_CACHE_PATH = add_prefix_path(file_path=os.getenv('HF_DATASETS_CACHE_PATH'), prefix_path=PREFIX_PATH)

PREPROCESSORS_PATH = add_prefix_path(file_path=os.getenv('PREPROCESSORS_PATH'), prefix_path=PREFIX_PATH)
MODELS_PATH = add_prefix_path(file_path=os.getenv('MODELS_PATH'), prefix_path=PREFIX_PATH)
PIPELINES_PATH = add_prefix_path(file_path=os.getenv('PIPELINES_PATH'), prefix_path=PREFIX_PATH)

BASELINE_MODEL_PATH = add_prefix_path(file_path=os.getenv('BASELINE_MODEL_PATH'), prefix_path=PREFIX_PATH)
TRANSFORMERS_MODEL_PATH = add_prefix_path(file_path=os.getenv('TRANSFORMERS_MODEL_PATH'), prefix_path=PREFIX_PATH)

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