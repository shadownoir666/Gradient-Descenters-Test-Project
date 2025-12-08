from typing import List
from pydantic import BaseModel

class Segment(BaseModel):
    start: float
from typing import List
from pydantic import BaseModel

class Segment(BaseModel):
    start: float
    end: float
    speaker: str
    text: str
    emotion: str | None = None
    emotion_score: float | None = None

class Dendrogram(BaseModel):
    name: str
    threshold: float
    score: float
    image: str  # Base64 encoded image

class AnalysisResult(BaseModel):
    audio_id: str
    segments: List[Segment]
    dendrograms: List[Dendrogram] = []