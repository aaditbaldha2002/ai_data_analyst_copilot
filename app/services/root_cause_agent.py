import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def _identify_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        try:
            pd.to_datetime(df[col])
            return col
        except (ValueError, TypeError):
            continue
    raise ValueError("No date column found for root cause analysis.")


def analyze_root_cause(result: list[dict], target_metric: str, split_date: str = None) -> dict:
    df = pd.DataFrame(result)
    date_col = _identify_date_column(df)
    target_col = target_metric

    if target_col not in df.columns:
        raise ValueError(f"Target metric '{target_col}' not found in query result columns.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    if len(df) < 10:
        return {"error": "Not enough data for root cause analysis.", "insights": None}

    midpoint = split_date or df[date_col].median()
    df["_period"] = np.where(df[date_col] < midpoint, "before", "after")

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    correlations = {}
    if numeric_cols:
        corr_series = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        correlations = corr_series.round(3).to_dict()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in (date_col, "_period")]
    group_shifts = {}
    for col in categorical_cols:
        before_avg = df[df["_period"] == "before"].groupby(col)[target_col].mean()
        after_avg = df[df["_period"] == "after"].groupby(col)[target_col].mean()
        shift = (after_avg - before_avg).dropna().sort_values()
        if not shift.empty:
            group_shifts[col] = shift.round(2).to_dict()

    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    X = X.fillna(0)
    y = df[target_col]

    shap_importance = {}
    try:
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = dict(zip(feature_cols, [round(float(v), 3) for v in mean_abs_shap]))
        shap_importance = dict(sorted(shap_importance.items(), key=lambda x: -x[1]))
    except Exception:
        shap_importance = {}

    return {
        "target_metric": target_col,
        "date_column": date_col,
        "split_point": str(midpoint),
        "correlations": correlations,
        "group_shifts": group_shifts,
        "feature_importance_shap": shap_importance,
        "error": None,
    }