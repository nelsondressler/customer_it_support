import os
import sys

import config

def add_prefix_path(file_path: str, prefix_path: str):
    return os.path.join(prefix_path, file_path)

def get_artifact_folder_path(artifact_type: str, model_type: str = None):
    if artifact_type == 'Preprocessor':
        return config.PREPROCESSOR_FOLDER_PATH
    elif artifact_type == 'Model':
        if artifact_type == 'Model' and model_type == 'Transformer':
            return config.TRANSFORMER_FOLDER_PATH
        elif artifact_type == 'Model' and model_type == 'Baseline':
            return config.MODEL_FOLDER_PATH
        else:
            return config.MODEL_FOLDER_PATH
    elif artifact_type == 'Pipeline':
        return config.PIPELINE_FOLDER_PATH