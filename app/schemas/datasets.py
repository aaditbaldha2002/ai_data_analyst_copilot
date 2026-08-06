from datetime import datetime
from typing import Any

from pydantic import BaseModel

class DatasetOut(BaseModel):
    id: int
    filename: str
    schema_json: dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True