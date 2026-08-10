from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class QueryResult(BaseModel):
    id: int
    question: str
    generated_sql: str
    result_json: Optional[list[dict[str, Any]]] = None
    error: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True