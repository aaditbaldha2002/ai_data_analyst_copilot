# ============================================================
# 3. FEATURE EXTRACTOR
# ============================================================

import uuid

import pandas as pd
from langgraph.runtime import Runtime

from app.repositories.dataset_repository import DatasetRepository, DatasetNotFoundError
from app.dataset_ingestion import DATA_ROOT
from app.services.graph_state import GraphState
from app.services.graph_context import GraphContext

# Scratch space for intermediate, run-specific artifacts (as opposed to
# DATA_ROOT/<dataset_id>/, which holds the canonical uploaded dataset).
# Each anomaly-detection run gets its own subfolder so concurrent runs
# on the same dataset never collide.
SCRATCH_ROOT = DATA_ROOT / "_scratch" / "anomaly_runs"

# A grouping column with more distinct values than this (as an absolute
# count, or as a fraction of total rows) stops being a meaningful
# segmentation dimension — it's effectively an ID. e.g. grouping 200 rows
# by a column with 200 unique values produces 200 groups of 1 row each.
MAX_GROUPING_CARDINALITY = 50
MAX_GROUPING_CARDINALITY_RATIO = 0.2


def _naive_feature_columns(profile: dict) -> list[str]:
    """Fallback used only when profile_decision is unavailable (e.g. the
    LLM decision step was skipped). Mirrors the pre-existing heuristic:
    numeric columns minus anything flagged as a likely ID."""
    numeric_columns = profile.get("numeric_columns", [])
    possible_id_columns = set(profile.get("possible_id_columns", []))
    return [c for c in numeric_columns if c not in possible_id_columns]


def _validate_feature_columns(columns: list[str], df, possible_id_columns: set[str]) -> list[str]:
    """
    A column existing in df isn't enough — it also has to actually be
    numeric (anomaly-detection models can't fit on strings/categoricals)
    and not be a likely ID column, even if the LLM decision picked it.
    Same principle as the schema-hallucination guards used elsewhere:
    never trust a selected column name without checking its properties,
    not just its existence.
    """
    valid = []
    for c in columns:
        if c not in df.columns:
            continue
        if c in possible_id_columns:
            continue
        # pd.api.types.is_numeric_dtype (not np.issubdtype) — the latter
        # can't interpret pandas 3.0's new string/StringDtype and throws,
        # same category of dtype-detection bug already found twice
        # elsewhere in this project.
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool:
            continue
        valid.append(c)
    return valid


def _validate_grouping_columns(columns: list[str], df) -> list[str]:
    """
    Existence isn't enough here either — a grouping column with near-1-row
    groups (i.e. close to unique per row) is functionally an ID and
    produces meaningless segmentation. Filtered by both an absolute
    cardinality cap and a ratio-to-row-count cap, since a small dataset
    with genuinely 40 distinct groups is fine, but a large dataset where
    40 distinct values is still 90% unique is not.
    """
    valid = []
    row_count = len(df)
    for c in columns:
        if c not in df.columns:
            continue
        distinct = df[c].nunique(dropna=True)
        if distinct > MAX_GROUPING_CARDINALITY:
            continue
        if row_count > 0 and (distinct / row_count) > MAX_GROUPING_CARDINALITY_RATIO:
            continue
        valid.append(c)
    return valid


def _validate_date_column(column: str | None, df) -> str | None:
    """Existence + actually-a-datetime-dtype check, same guard pattern as
    feature/grouping column validation."""
    if column is None:
        return None
    if column not in df.columns:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        return None
    return column


def feature_extractor(graph: GraphState, runtime: Runtime[GraphContext]) -> dict:
    """
    Selects the feature columns, grouping columns, and (if present) the
    time-axis date column for anomaly detection, and writes the pruned
    matrix to a scratch Parquet file.

    Column selection:
        - Primary source: profile_decision.final_target_columns /
          final_grouping_columns / date_column (the LLM's grounded
          decision from data_profiler).
        - Fallback: naive numeric-minus-ID heuristic for feature columns
          only, used if profile_decision is missing. Grouping and
          date_column have no naive fallback — they're decision-only.

    Missing-value handling is deliberately NOT done here — the matrix is
    written raw. feature_transformer (next node) owns imputation/scaling,
    including time-aware imputation when date_column is present.

    Only feature_run_id, features_path, feature_columns, grouping_columns,
    and date_column travel through graph state — never the dataframe
    itself. Downstream nodes read the same Parquet file rather than
    re-deriving column selection, which keeps every node working off
    identical features and avoids recomputation.
    """
    dataset_id = graph.get("dataset_id")
    if not dataset_id:
        return {"error": "No dataset_id present in graph state."}

    try:
        db = runtime.context.db
        repo = DatasetRepository(db)
        try:
            df = repo.get_dataframe(dataset_id)
        except DatasetNotFoundError:
            return {"error": f"Could not find processed data for dataset_id={dataset_id}."}

        if df.empty:
            return {
                "feature_columns": [],
                "grouping_columns": [],
                "date_column": None,
                "features_path": None,
                "warning": "No valid dataframe available for feature extraction.",
            }

        profile = graph.get("data_profile", {}) or {}
        decision = graph.get("profile_decision") or {}
        possible_id_columns = set(profile.get("possible_id_columns", []))

        # Computed independently of each other — a decision with valid
        # grouping columns but an empty/missing target-column list
        # shouldn't silently lose its grouping columns just because the
        # target side came up empty.
        raw_target_columns = list(dict.fromkeys(decision.get("final_target_columns", [])))
        raw_grouping_columns = list(dict.fromkeys(decision.get("final_grouping_columns", [])))

        feature_columns = _validate_feature_columns(raw_target_columns, df, possible_id_columns)
        grouping_columns = _validate_grouping_columns(raw_grouping_columns, df)
        date_column = _validate_date_column(decision.get("date_column"), df)

        # Fall back to the naive heuristic if profile_decision was
        # missing OR if everything it picked got filtered out by
        # validation (e.g. the LLM's only suggested column turned out to
        # be non-numeric) — an empty decision-driven result shouldn't be
        # a hard failure when a reasonable heuristic fallback exists.
        if not feature_columns:
            feature_columns = _validate_feature_columns(
                _naive_feature_columns(profile), df, possible_id_columns
            )

        if not feature_columns:
            return {
                "feature_columns": [],
                "grouping_columns": grouping_columns,
                "date_column": date_column,
                "features_path": None,
                "warning": "No suitable numerical features were found for anomaly detection.",
            }

        columns_to_write = feature_columns + [
            c for c in grouping_columns if c not in feature_columns
        ]
        if date_column and date_column not in columns_to_write:
            columns_to_write.append(date_column)

        selected = df[columns_to_write].copy()

        run_id = uuid.uuid4().hex
        run_dir = SCRATCH_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        features_path = run_dir / "features.parquet"
        selected.to_parquet(features_path, engine="pyarrow", index=False)

        return {
            "feature_run_id": run_id,
            "features_path": str(features_path),
            "feature_columns": feature_columns,
            "grouping_columns": grouping_columns,
            "date_column": date_column,
        }
    except Exception as e:
        return {"error": str(e)}