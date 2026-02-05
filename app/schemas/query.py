from typing import Optional, Dict, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    user_question: str = Field(..., min_length=5)
    filters: Optional[Dict[str, str]] = None


class SourceRef(BaseModel):
    source: str
    page: Optional[int]


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    status: str
    sources: List[SourceRef]


class ErrorResponse(BaseModel):
    detail: str
