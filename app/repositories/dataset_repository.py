"""
app/repositories/dataset_repository.py

The single point of access every agent (SQL Agent, Forecast Agent,
Regression Agent, Classification Agent, Anomaly Agent, Root Cause Agent)
goes through to read a dataset. Nobody else should touch Postgres'
dataset_columns table or the Parquet files on disk directly — that's how
schema drift between agents gets prevented (see the DatasetSchema/kind
system built in dataset_ingestion.py).

Usage in a FastAPI route:
    def some_route(db: Session = Depends(get_db)):
        repo = DatasetRepository(db)
        df = repo.get_dataframe(dataset_id, columns=["date", "revenue"])

Usage in a LangGraph node (outside request scope):
    from app.database import SessionLocal
    def forecast_node(state):
        with SessionLocal() as db:
            repo = DatasetRepository(db)
            date_col = repo.get_date_column(state["dataset_id"])
            df = repo.get_dataframe(state["dataset_id"], columns=[date_col, state["target"]])
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from sqlalchemy.orm import Session

from app.dataset_ingestion import ColumnKind, ColumnSchema, DatasetSchema
from app.models.datasets import Dataset, DatasetColumn


class DatasetNotFoundError(Exception):
    def __init__(self, dataset_id: int):
        super().__init__(f"Dataset {dataset_id} not found or has no parquet_path set")
        self.dataset_id = dataset_id


class DatasetRepository:
    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self, dataset_id: int) -> DatasetSchema:
        """
        Builds the schema manifest from Postgres (Dataset + DatasetColumn
        rows) — this is the ONE place agents should learn about a
        dataset's columns. Never re-infer schema from the Parquet file
        directly; that's what causes drift between agents.
        """
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None or not dataset.parquet_path:
            raise DatasetNotFoundError(dataset_id)

        column_rows = (
            self.db.query(DatasetColumn)
            .filter(DatasetColumn.dataset_id == dataset_id)
            .order_by(DatasetColumn.id)
            .all()
        )

        columns = [
            ColumnSchema(
                name=c.name,
                dtype=c.dtype,
                kind=c.kind,
                null_pct=c.null_pct,
                cardinality=c.cardinality,
                sample_values=c.sample_values or [],
            )
            for c in column_rows
        ]

        return DatasetSchema(
            dataset_id=str(dataset.id),
            row_count=dataset.row_count or 0,
            columns=columns,
            content_hash=dataset.content_hash or "",
            converted_at=dataset.created_at.isoformat() if dataset.created_at else "",
        )

    def columns_by_kind(self, dataset_id: int, kind: ColumnKind) -> list[str]:
        """
        SQL-side filter — e.g. columns_by_kind(19, "numeric") for a
        Regression Agent picking candidate features. Runs as a targeted
        query, not a fetch-everything-then-filter-in-Python pass.
        """
        rows = (
            self.db.query(DatasetColumn.name)
            .filter(DatasetColumn.dataset_id == dataset_id, DatasetColumn.kind == kind)
            .all()
        )
        return [r.name for r in rows]

    def get_date_column(self, dataset_id: int) -> str | None:
        """Convenience for Forecast/Anomaly agents. Returns the first
        date-kind column, or None if the dataset has no date column."""
        cols = self.columns_by_kind(dataset_id, "date")
        return cols[0] if cols else None

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _parquet_path(self, dataset_id: int) -> str:
        dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset is None or not dataset.parquet_path:
            raise DatasetNotFoundError(dataset_id)
        return dataset.parquet_path

    def get_dataframe(
        self, dataset_id: int, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Column-pruned Parquet read. Pass only the columns the calling
        agent actually needs — pruning happens at the parquet scan level,
        so this stays cheap even on wide datasets.
        """
        path = self._parquet_path(dataset_id)
        return pd.read_parquet(path, columns=columns)

    def get_duckdb_relation(
        self, dataset_id: int, con: duckdb.DuckDBPyConnection
    ) -> duckdb.DuckDBPyRelation:
        """
        For the SQL Agent — registers a relation over the Parquet file
        directly. No copy into memory, no separate persistent .duckdb
        file to keep in sync; DuckDB reads the same Parquet the ML
        agents read.
        """
        path = self._parquet_path(dataset_id)
        # DuckDB needs forward slashes even on Windows paths
        posix_path = Path(path).as_posix()
        return con.sql(f"SELECT * FROM read_parquet('{posix_path}')")