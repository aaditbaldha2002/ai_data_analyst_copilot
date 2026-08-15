from datetime import datetime
from typing import Any

from pydantic import BaseModel


class DatasetColumnOut(BaseModel):
    name: str
    dtype: str
    kind: str
    null_pct: float
    cardinality: int | None = None
    sample_values: list[Any] | None = None

    class Config:
        from_attributes = True


class DatasetOut(BaseModel):
    id: int
    filename: str
    row_count: int | None = None
    content_hash: str | None = None
    columns: list[DatasetColumnOut] = []
    created_at: datetime

    class Config:
        from_attributes = True