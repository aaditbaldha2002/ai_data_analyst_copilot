"""
dataset_ingestion.py

Converts uploaded raw files (CSV / Excel) into a canonical, typed Parquet
file plus a schema manifest. This is the ONE place schema inference and
date-detection happen — every agent downstream (SQL, Forecast, Anomaly,
Regression, Classification) reads the Parquet + manifest instead of
re-inferring anything.

Integration points (marked TODO) are where this hooks into your existing
FastAPI upload endpoint and `datasets` Postgres model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# --------------------------------------------------------------------------
# Config — adjust root to match your existing local-disk layout
# --------------------------------------------------------------------------

DATA_ROOT = Path(os.environ.get("DATASET_STORAGE_ROOT", "/data"))

DATE_NAME_HINTS = re.compile(
    r"(date|time|timestamp|created|updated|_at$|_dt$|day|month|year)", re.IGNORECASE
)

# Years outside this range are treated as parse artifacts, not real dates
# (e.g. "01/02" parsing as year 1). Adjust if you genuinely expect
# historical/far-future dates in your datasets.
MIN_PLAUSIBLE_YEAR = 1900
MAX_PLAUSIBLE_YEAR = 2100

ColumnKind = Literal["numeric", "date", "categorical", "boolean", "text", "id"]


# --------------------------------------------------------------------------
# Schema manifest data structures
# --------------------------------------------------------------------------

@dataclass
class ColumnSchema:
    name: str
    dtype: str            # pandas dtype string, e.g. "int64", "float64", "datetime64[ns]"
    kind: ColumnKind       # semantic kind used by agents to pick columns
    null_pct: float
    cardinality: int | None = None       # distinct value count, useful for categoricals
    sample_values: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "kind": self.kind,
            "null_pct": round(self.null_pct, 4),
            "cardinality": self.cardinality,
            "sample_values": self.sample_values,
        }


@dataclass
class DatasetSchema:
    dataset_id: str
    row_count: int
    columns: list[ColumnSchema]
    content_hash: str
    converted_at: str

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "content_hash": self.content_hash,
            "converted_at": self.converted_at,
            "columns": [c.to_dict() for c in self.columns],
        }

    def column_names(self, kind: ColumnKind | None = None) -> list[str]:
        if kind is None:
            return [c.name for c in self.columns]
        return [c.name for c in self.columns if c.kind == kind]


# --------------------------------------------------------------------------
# Date detection — same principle as your existing _looks_like_date helper,
# centralized here so ingestion and every agent agree on what "is a date"
# --------------------------------------------------------------------------

def _looks_like_date(series: pd.Series, col_name: str) -> bool:
    """
    Three-signal check: dtype, column name, and a sample-parse attempt.
    Mirrors the guard already used in the segmentation/grouping code so
    date columns never get treated as categorical/grouping dimensions
    anywhere in the system.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # pandas >= 2.x (and especially 3.0's new default string dtype) means
    # "is this a string column" is no longer reliably `dtype == object`.
    # is_string_dtype covers legacy object-string columns AND the new
    # pandas StringDtype/"str" dtype.
    if pd.api.types.is_string_dtype(series) and not pd.api.types.is_numeric_dtype(series):
        name_hint = bool(DATE_NAME_HINTS.search(col_name))
        sample = series.dropna().head(25)
        if len(sample) == 0:
            return False
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except Exception:
            return False

        # Reject parses that land outside a plausible calendar range.
        # pd.to_datetime happily accepts short/ambiguous strings like
        # "01/02" as year 1 (or "99" as 1999, etc.) with format="mixed" —
        # those aren't real dates, they're short codes or fragments that
        # coincidentally parse. Only count a parse as a genuine date hit
        # if the year falls in a sane range.
        in_range = parsed.dt.year.between(MIN_PLAUSIBLE_YEAR, MAX_PLAUSIBLE_YEAR)
        valid_parsed = parsed.where(in_range)
        parse_rate = valid_parsed.notna().mean()

        # Require a strong parse rate; only need the name hint to lower
        # the bar slightly (e.g. ambiguous "01/02/2024"-style columns)
        threshold = 0.8 if not name_hint else 0.6
        return parse_rate >= threshold

    return False


def _infer_column_kind(series: pd.Series, col_name: str) -> ColumnKind:
    if _looks_like_date(series, col_name):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # object/string columns: id vs categorical vs free text, by cardinality
    non_null = series.dropna()
    if len(non_null) == 0:
        return "text"
    distinct_ratio = non_null.nunique() / max(len(non_null), 1)
    if distinct_ratio > 0.95 and non_null.nunique() > 50:
        return "id"
    if non_null.nunique() <= max(50, int(len(non_null) * 0.2)):
        return "categorical"
    return "text"


# --------------------------------------------------------------------------
# Core conversion
# --------------------------------------------------------------------------

def _read_raw(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _apply_typed_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, list[ColumnSchema]]:
    columns: list[ColumnSchema] = []

    for col in df.columns:
        series = df[col]
        kind = _infer_column_kind(series, col)

        if kind == "date" and not pd.api.types.is_datetime64_any_dtype(series):
            df[col] = pd.to_datetime(series, errors="coerce", format="mixed")
            series = df[col]

        null_pct = series.isna().mean()
        cardinality = int(series.nunique(dropna=True)) if kind in ("categorical", "id", "boolean") else None
        sample_values = (
            series.dropna().astype(str).unique()[:5].tolist() if kind == "categorical" else []
        )

        columns.append(
            ColumnSchema(
                name=col,
                dtype=str(series.dtype),
                kind=kind,
                null_pct=float(null_pct),
                cardinality=cardinality,
                sample_values=sample_values,
            )
        )

    return df, columns


def _content_hash(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_to_parquet(dataset_id: str, raw_file_path: str | Path) -> DatasetSchema:
    """
    Main entry point. Call this once at upload time.

    - dataset_id: the ID already generated for the `datasets` Postgres row
      (TODO: match whatever ID scheme your upload endpoint currently uses)
    - raw_file_path: path to the just-uploaded CSV/Excel file on local disk

    Returns a DatasetSchema — persist schema.to_dict() to Postgres
    (either as a JSON column on `datasets`, or normalized into a
    `dataset_columns` table, one row per ColumnSchema).
    """
    raw_file_path = Path(raw_file_path)
    dataset_dir = DATA_ROOT / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _content_hash(raw_file_path)

    df = _read_raw(raw_file_path)
    df, columns = _apply_typed_schema(df)

    parquet_path = dataset_dir / "data.parquet"
    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    # Keep the original file for audit/re-import purposes
    raw_copy_path = dataset_dir / f"raw{raw_file_path.suffix.lower()}"
    if raw_file_path.resolve() != raw_copy_path.resolve():
        raw_copy_path.write_bytes(raw_file_path.read_bytes())

    schema = DatasetSchema(
        dataset_id=dataset_id,
        row_count=len(df),
        columns=columns,
        content_hash=content_hash,
        converted_at=datetime.utcnow().isoformat(),
    )

    # Also drop the manifest as JSON next to the parquet file — cheap,
    # human-inspectable, and a fallback if the Postgres write fails
    # partway (re-sync on next read rather than losing the schema).
    (dataset_dir / "schema.json").write_text(json.dumps(schema.to_dict(), indent=2))

    return schema


# --------------------------------------------------------------------------
# ORM row builders — call these from the FastAPI endpoint after
# convert_to_parquet() returns. Kept here (not in the router) so the
# mapping from ColumnSchema -> DB row lives next to the schema definition.
# --------------------------------------------------------------------------

def build_dataset_column_rows(schema: DatasetSchema, dataset_pk: int, DatasetColumn) -> list:
    """
    OPTION B (normalized table). Returns a list of DatasetColumn ORM
    instances ready to db.add_all(). `DatasetColumn` is passed in rather
    than imported here to avoid this module depending on your app package.
    """
    return [
        DatasetColumn(
            dataset_id=dataset_pk,
            name=c.name,
            dtype=c.dtype,
            kind=c.kind,
            null_pct=c.null_pct,
            cardinality=c.cardinality,
            sample_values=c.sample_values or None,
        )
        for c in schema.columns
    ]


def build_schema_json(schema: DatasetSchema) -> dict:
    """OPTION A (JSONB column). Just the manifest dict, ready to assign
    directly to dataset.schema_json."""
    return schema.to_dict()


# --------------------------------------------------------------------------
# Example FastAPI integration (TODO: adapt to your actual upload endpoint)
# --------------------------------------------------------------------------
#
# @router.post("/datasets/upload")
# async def upload_dataset(file: UploadFile, db: Session = Depends(get_db)):
#     dataset_id = str(uuid4())
#     tmp_path = Path(f"/tmp/{dataset_id}_{file.filename}")
#     tmp_path.write_bytes(await file.read())
#
#     schema = convert_to_parquet(dataset_id, tmp_path)
#
#     dataset_row = Dataset(
#         id=dataset_id,
#         filename=file.filename,
#         row_count=schema.row_count,
#         content_hash=schema.content_hash,
#         schema_json=schema.to_dict(),   # JSONB column
#         created_at=datetime.utcnow(),
#     )
#     db.add(dataset_row)
#     db.commit()
#
#     tmp_path.unlink(missing_ok=True)
#     return {"dataset_id": dataset_id, "schema": schema.to_dict()}