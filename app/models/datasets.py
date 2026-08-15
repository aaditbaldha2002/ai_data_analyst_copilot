from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)

    # Raw upload, kept for audit/re-import
    raw_file_path = Column(String, nullable=False)

    # Canonical typed copy every agent reads from
    parquet_path = Column(String, nullable=True)

    row_count = Column(Integer, nullable=True)
    content_hash = Column(String, nullable=True, index=True)

    # --- OPTION A (JSONB alternative) ---
    # If you go this route instead of dataset_columns, keep this and skip
    # the DatasetColumn model + relationship below.
    # schema_json = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    columns = relationship(
        "DatasetColumn", back_populates="dataset", cascade="all, delete-orphan"
    )


class DatasetColumn(Base):
    """
    OPTION B (recommended): one row per column, so agents can filter by
    kind directly in SQL instead of fetching+parsing a JSON blob.

    e.g. SELECT name FROM dataset_columns WHERE dataset_id = :id AND kind = 'numeric'
    """
    __tablename__ = "dataset_columns"
    __table_args__ = (UniqueConstraint("dataset_id", "name", name="uq_dataset_column"),)

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True)

    name = Column(String, nullable=False)
    dtype = Column(String, nullable=False)          # pandas dtype string
    kind = Column(String, nullable=False, index=True)  # numeric | date | categorical | boolean | id | text
    null_pct = Column(Float, nullable=False, default=0.0)
    cardinality = Column(Integer, nullable=True)
    sample_values = Column(JSON, nullable=True)     # small list, JSON is fine here

    dataset = relationship("Dataset", back_populates="columns")