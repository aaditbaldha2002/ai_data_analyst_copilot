"""drop unused schema_json column from datasets (superseded by dataset_columns table)

Revision ID: drop_schema_json
Revises: add_dataset_columns
Create Date: 2026-08-15
"""
from alembic import op
import sqlalchemy as sa

revision = "drop_schema_json"
down_revision = "add_dataset_columns"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_column("datasets", "schema_json")


def downgrade():
    # Restored as nullable — the NOT NULL constraint can't be safely
    # reapplied without backfilling data for every existing row, so if
    # you ever downgrade, you'll need to backfill schema_json manually
    # before re-adding NOT NULL if you want it back.
    op.add_column("datasets", sa.Column("schema_json", sa.JSON(), nullable=True))