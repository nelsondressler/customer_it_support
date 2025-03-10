import pandas as pd
from pydantic import BaseModel

class ModelInput(BaseModel):
    model_choice: str

class PipelineInput(BaseModel):
    step_names: list

class FitModelResponse(BaseModel):
    model_path: str

class FitPipelineResponse(BaseModel):
    pipeline_path: str

class EmailInput(BaseModel):
    subject: str
    body: str
    model_choice: str  # e.g., "nb", "lr", "distilbert", "bert"

class PredictionResponse(BaseModel):
    queue: str
    priority: str
    details: dict    
    