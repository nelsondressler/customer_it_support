from pydantic import BaseModel

class EmailInput(BaseModel):
    subject: str
    body: str
    model_choice: str  # e.g., "nb", "lr", "distilbert", "bert"

class PredictionResponse(BaseModel):
    queue: str
    priority: str
    details: dict