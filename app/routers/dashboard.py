import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.security import get_current_user
from app.models.users import User
from app.models.datasets import Dataset
from app.services.duckdb_engine import run_query
from app.services.dashboard_agent import compute_kpis, segment_entities
from app.services.chart_renderer import render_chart
from app.services.report_generator import generate_dashboard_report
import pandas as pd

router = APIRouter(prefix="/datasets", tags=["dashboard"])


@router.post("/{dataset_id}/dashboard")
def generate_dashboard(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.owner_id == current_user.id)
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    ext = os.path.splitext(dataset.file_path)[1].lower()
    result = run_query(dataset.file_path, ext, "SELECT * FROM data")
    df = pd.DataFrame(result)

    kpis = compute_kpis(df)
    segmentation = segment_entities(df)

    chart_paths = []

    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    revenue_col = next((c for c in df.columns if "revenue" in c.lower()), None)
    if date_col and revenue_col:
        trend_df = df.copy()
        trend_df[date_col] = pd.to_datetime(trend_df[date_col]).dt.strftime("%Y-%m")
        trend_agg = trend_df.groupby(date_col)[revenue_col].sum().reset_index()
        trend_chart_config = {
            "chart_type": "line", "x_key": date_col, "y_key": revenue_col,
            "title": f"{revenue_col.title()} Trend",
        }
        path = render_chart(trend_chart_config, trend_agg.to_dict(orient="records"))
        if path:
            chart_paths.append(path)

    category_col = next((c for c in df.columns if c in ("category", "product", "region")), None)
    if category_col and revenue_col:
        cat_agg = df.groupby(category_col)[revenue_col].sum().reset_index()
        cat_chart_config = {
            "chart_type": "bar", "x_key": category_col, "y_key": revenue_col,
            "title": f"{revenue_col.title()} by {category_col.title()}",
        }
        path = render_chart(cat_chart_config, cat_agg.to_dict(orient="records"))
        if path:
            chart_paths.append(path)

    report_path = generate_dashboard_report(dataset.filename, kpis, segmentation, chart_paths)

    return {
        "kpis": kpis,
        "segmentation": segmentation,
        "chart_paths": chart_paths,
        "report_path": report_path,
    }