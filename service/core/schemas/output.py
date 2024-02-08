from pydantic import BaseModel

class APIOutput(BaseModel):
    Emotion: str
    TimeElapsed: str