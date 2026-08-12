import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _get_date_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            pd.to_datetime(df[col])
            return col
        except (ValueError, TypeError):
            continue
    return None


def _looks_like_date(df: pd.DataFrame, col: str) -> bool:
    """True if a column is a date/datetime, whether typed as datetime64 or a plain string."""
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    if "date" in col.lower() or "time" in col.lower():
        return True
    try:
        sample = df[col].dropna().astype(str).iloc[:5]
        if sample.empty:
            return False
        pd.to_datetime(sample, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def compute_kpis(df: pd.DataFrame) -> dict:
    numeric_cols = _get_numeric_columns(df)
    date_col = _get_date_column(df)

    kpis = {"total_rows": len(df)}

    revenue_col = next((c for c in numeric_cols if "revenue" in c.lower()), None) or (
        numeric_cols[0] if numeric_cols else None
    )

    if revenue_col:
        kpis["total_" + revenue_col] = round(float(df[revenue_col].sum()), 2)
        kpis["average_" + revenue_col] = round(float(df[revenue_col].mean()), 2)

    units_col = next((c for c in numeric_cols if "unit" in c.lower() and "price" not in c.lower()), None)
    if units_col:
        kpis["total_" + units_col] = int(df[units_col].sum())

    if date_col and revenue_col:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        midpoint = df[date_col].median()
        before = df[df[date_col] < midpoint][revenue_col].sum()
        after = df[df[date_col] >= midpoint][revenue_col].sum()
        if before > 0:
            growth_pct = round(((after - before) / before) * 100, 1)
            kpis["period_over_period_growth_pct"] = growth_pct

    return kpis


def segment_entities(df: pd.DataFrame, group_col_candidates: list[str] = None) -> dict:
    """
    Groups rows by a meaningful categorical column (e.g. product, category, region)
    and clusters those groups by their aggregate numeric behavior (KMeans),
    returning labeled segments (e.g. "High performers", "Underperformers").

    Date-like columns are explicitly excluded as grouping candidates, even if
    pandas has typed them as object/string, since a date is never a meaningful
    segmentation dimension for this purpose.
    """
    numeric_cols = _get_numeric_columns(df)
    all_categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Exclude anything that looks like a date, regardless of dtype
    categorical_cols = [c for c in all_categorical_cols if not _looks_like_date(df, c)]

    if not categorical_cols or not numeric_cols:
        return {
            "error": "Not enough categorical/numeric data to build segments.",
            "group_column": None,
            "segments": [],
        }

    # Prefer well-known, meaningful business dimensions if present; otherwise
    # fall back to whatever non-date categorical columns remain.
    preferred_order = ["product", "category", "region", "customer", "customer_id", "segment"]
    ordered_candidates = group_col_candidates or (
        [c for c in preferred_order if c in categorical_cols]
        + [c for c in categorical_cols if c not in preferred_order]
    )
    group_col = ordered_candidates[0] if ordered_candidates else categorical_cols[0]

    agg = df.groupby(group_col)[numeric_cols].agg(["sum", "mean", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()

    if len(agg) < 3:
        return {
            "error": f"Not enough distinct '{group_col}' values to segment (need at least 3, found {len(agg)}).",
            "group_column": group_col,
            "segments": [],
        }

    feature_cols = [c for c in agg.columns if c != group_col]
    X = StandardScaler().fit_transform(agg[feature_cols].fillna(0))

    k = min(3, len(agg))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["cluster"] = kmeans.fit_predict(X)

    # Rank clusters by their primary revenue-like metric to assign readable labels.
    revenue_like_candidates = [c for c in feature_cols if "revenue" in c.lower() and c.endswith("_sum")]
    rank_metric = revenue_like_candidates[0] if revenue_like_candidates else feature_cols[0]

    cluster_rank = agg.groupby("cluster")[rank_metric].mean().sort_values(ascending=False)
    label_pool = ["High performers", "Steady performers", "Underperformers"]
    labels = label_pool[:k]
    label_map = {cluster_id: labels[i] for i, cluster_id in enumerate(cluster_rank.index)}
    agg["segment_label"] = agg["cluster"].map(label_map)

    segments = []
    for label in dict.fromkeys(label_map.values()):
        members = agg.loc[agg["segment_label"] == label, group_col].tolist()
        segments.append({"segment": label, group_col: members})

    return {"group_column": group_col, "segments": segments, "error": None}