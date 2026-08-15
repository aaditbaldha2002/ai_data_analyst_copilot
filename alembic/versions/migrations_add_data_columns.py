"""add parquet_path, content_hash, row_count to datasets; add dataset_columns table

Revision ID: add_dataset_columns
Revises: 793edae8625a
Create Date: 2026-08-14
"""
from alembic import op
import sqlalchemy as sa

revision = "add_dataset_columns"
down_revision = "793edae8625a"
branch_labels = None
depends_on = None


def upgrade():
    # --- datasets table changes ---
    op.alter_column("datasets", "file_path", new_column_name="raw_file_path")
    op.add_column("datasets", sa.Column("parquet_path", sa.String(), nullable=True))
    op.add_column("datasets", sa.Column("row_count", sa.Integer(), nullable=True))
    op.add_column("datasets", sa.Column("content_hash", sa.String(), nullable=True))
    op.create_index("ix_datasets_content_hash", "datasets", ["content_hash"])

    # If you're keeping schema_json for a transition period, leave it —
    # otherwise: op.drop_column("datasets", "schema_json")

    # --- new dataset_columns table ---
    op.create_table(
        "dataset_columns",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column(
            "dataset_id",
            sa.Integer(),
            sa.ForeignKey("datasets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("dtype", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("null_pct", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cardinality", sa.Integer(), nullable=True),
        sa.Column("sample_values", sa.JSON(), nullable=True),
        sa.UniqueConstraint("dataset_id", "name", name="uq_dataset_column"),
    )
    op.create_index("ix_dataset_columns_dataset_id", "dataset_columns", ["dataset_id"])
    op.create_index("ix_dataset_columns_kind", "dataset_columns", ["kind"])


def downgrade():
    op.drop_index("ix_dataset_columns_kind", table_name="dataset_columns")
    op.drop_index("ix_dataset_columns_dataset_id", table_name="dataset_columns")
    op.drop_table("dataset_columns")

    op.drop_index("ix_datasets_content_hash", table_name="datasets")
    op.drop_column("datasets", "content_hash")
    op.drop_column("datasets", "row_count")
    op.drop_column("datasets", "parquet_path")
    op.alter_column("datasets", "raw_file_path", new_column_name="file_path")