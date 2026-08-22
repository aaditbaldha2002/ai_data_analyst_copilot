# ============================================================
# 4. FEATURE TRANSFORMER
# ============================================================

import uuid

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app.dataset_ingestion import DATA_ROOT
from app.services.graph_state import GraphState

SCRATCH_ROOT = DATA_ROOT / "_scratch" / "anomaly_runs"

# Below this many rows, a group can't support its own meaningful median/
# scaler (e.g. a 1-2 row group has near-zero variance by construction,
# and a single-row group's "median" is just that one value). Groups
# smaller than this fall back to the global statistic instead.
MIN_GROUP_SIZE_FOR_PER_GROUP_STATS = 5

# Sentinel used as the dict key for the "no grouping" case, so the same
# per-group loop logic handles both grouped and ungrouped data without
# two separate code paths.
_GLOBAL_GROUP_KEY = "__global__"


def _impute_series(series: pd.Series, date_series: pd.Series | None) -> pd.Series:
    """
    Fills missing values in a single column, within a single group.

    If a date axis is available: sorts by date, linearly interpolates
    between known points, then forward/backward-fills any leading or
    trailing gaps interpolation can't reach (interpolate() only fills
    *between* two known values by default — limit_direction='both'
    additionally covers gaps at the start/end of the series).

    If no date axis (or too few points to interpolate meaningfully):
    falls back to a flat median fill, same as before this change.
    """
    if series.isna().all():
        return series.fillna(0.0)

    if date_series is not None and len(series) >= 2:
        order = date_series.sort_values().index
        sorted_series = series.loc[order]
        filled = sorted_series.interpolate(method="linear", limit_direction="both")
        return filled.reindex(series.index)

    median = series.median()
    if pd.isna(median):
        median = 0.0
    return series.fillna(median)


def _median_fallback(series: pd.Series) -> float:
    """Per-group (or global) median, used for the persisted inference-time
    fallback stat — even when this batch used time-aware interpolation to
    fit, a single new row scored later has no neighbors to interpolate
    against, so a plain median fallback is still needed for that case."""
    median = series.median()
    return float(median) if pd.notna(median) else 0.0


def feature_transformer(graph: GraphState) -> dict:
    """
    Imputes missing values and scales the feature matrix written by
    feature_extractor, producing a model-ready transformed matrix.

    Per-group behavior (per your grouping_columns): each group gets its
    own median/scaler, EXCEPT groups smaller than
    MIN_GROUP_SIZE_FOR_PER_GROUP_STATS, which fall back to the global
    median/scaler — a 2-row group can't support a meaningful scaler of
    its own. With no grouping_columns, everything is treated as a single
    global group.

    Imputation is time-aware when date_column is present: values are
    filled via linear interpolation along the sorted time axis (plus
    edge fill for leading/trailing gaps) rather than a flat statistic,
    since a flat median/mean ignores trend and seasonality and can
    silently introduce fake anomalies at fill points. Falls back to
    median fill when there's no date axis.

    Outputs (all JSON-safe — no live objects in graph state):
        - transformed_features_path: Parquet with scaled feature columns
          + untouched grouping_columns + untouched date_column, so rows
          stay traceable to their group/time for later nodes.
        - transform_artifacts_path: joblib file containing, per group,
          the fitted StandardScaler and the median fallback stats needed
          for consistent single-row inference later — persisted per your
          own docstring's original intent, rather than living in state.
    """
    features_path = graph.get("features_path")
    feature_columns = graph.get("feature_columns") or []
    grouping_columns = graph.get("grouping_columns") or []
    date_column = graph.get("date_column")
    run_id = graph.get("feature_run_id")

    if not features_path or not feature_columns:
        return {
            "transformed_features_path": None,
            "transform_artifacts_path": None,
            "warning": "No features available for transformation.",
        }

    try:
        df = pd.read_parquet(features_path)
        if df.empty:
            return {
                "transformed_features_path": None,
                "transform_artifacts_path": None,
                "warning": "Feature matrix is empty.",
            }

        date_series_full = df[date_column] if date_column else None

        # Compute GLOBAL stats first — used both as the ungrouped case
        # and as the fallback for undersized groups.
        global_medians = {c: _median_fallback(df[c]) for c in feature_columns}
        global_scaler = StandardScaler()
        global_imputed = df[feature_columns].copy()
        for c in feature_columns:
            global_imputed[c] = _impute_series(df[c], date_series_full)
        global_scaler.fit(global_imputed)

        artifacts = {
            "feature_columns": feature_columns,
            "grouping_columns": grouping_columns,
            "date_column": date_column,
            "groups": {},  # group_key -> {"scaler": ..., "medians": {...}, "row_count": N}
        }

        transformed = df.copy()

        if grouping_columns:
            group_iter = df.groupby(grouping_columns, dropna=False).groups.items()
        else:
            group_iter = [(_GLOBAL_GROUP_KEY, df.index)]

        for group_key, row_index in group_iter:
            group_df = df.loc[row_index]
            group_date_series = group_df[date_column] if date_column else None
            use_global = len(group_df) < MIN_GROUP_SIZE_FOR_PER_GROUP_STATS

            if use_global:
                scaler = global_scaler
                medians = global_medians
            else:
                imputed = group_df[feature_columns].copy()
                for c in feature_columns:
                    imputed[c] = _impute_series(group_df[c], group_date_series)
                scaler = StandardScaler()
                scaler.fit(imputed)
                medians = {c: _median_fallback(group_df[c]) for c in feature_columns}

            # Re-impute this group's rows (even undersized ones use their
            # OWN values for imputation — only the fitted scaler/medians
            # fall back to global; the actual fill values stay local to
            # what data the group has, since even a 2-row group's own
            # values are more relevant than the whole dataset's).
            imputed_for_transform = group_df[feature_columns].copy()
            for c in feature_columns:
                imputed_for_transform[c] = _impute_series(group_df[c], group_date_series)

            scaled_values = scaler.transform(imputed_for_transform)
            transformed.loc[row_index, feature_columns] = scaled_values

            artifacts["groups"][group_key] = {
                "scaler": scaler,
                "medians": medians,
                "row_count": len(group_df),
                "used_global_fallback": use_global,
            }

        run_id = run_id or uuid.uuid4().hex
        run_dir = SCRATCH_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        transformed_path = run_dir / "transformed_features.parquet"
        transformed.to_parquet(transformed_path, engine="pyarrow", index=False)

        artifacts_path = run_dir / "transform_artifacts.pkl"
        joblib.dump(artifacts, artifacts_path)

        return {
            "transformed_features_path": str(transformed_path),
            "transform_artifacts_path": str(artifacts_path),
        }
    except Exception as e:
        return {"error": str(e)}