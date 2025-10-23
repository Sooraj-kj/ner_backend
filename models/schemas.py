from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class MedicalEntity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float

class TranscriptionResponse(BaseModel):
    text: str
    is_final: bool
    language: Optional[str] = None
    timestamp: datetime
    entities: List[MedicalEntity] = []

class AudioConfig(BaseModel):
    sample_rate: int = 16000
    language: Optional[str] = None