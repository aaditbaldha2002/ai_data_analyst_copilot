import os
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.security import get_current_user
from app.models.users import User
from app.models.datasets import Dataset
from app.models.queries import Query
from app.schemas.queries import QueryRequest, QueryResult
from app.services.sql_agent import generate_sql
from app.services.duckdb_engine import run_query
from app.services.chart_agent import generate_chart_config
from app.services.explanation_agent import generate_explanation
from app.services.chart_renderer import render_chart

router = APIRouter(prefix="/datasets", tags=["queries"])


@router.post("/{dataset_id}/query", response_model=QueryResult)
def query_dataset(
    dataset_id: int,
    request: QueryRequest,
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

    sql = generate_sql(request.question, dataset.schema_json)

    error = None
    result = None
    chart_config = None
    chart_image_path = None
    explanation = None

    try:
        ext = os.path.splitext(dataset.file_path)[1].lower()
        result = run_query(dataset.file_path, ext, sql)

        chart_config = generate_chart_config(request.question, result)
        chart_image_path = render_chart(chart_config, result)
        explanation = generate_explanation(request.question, result)

    except Exception as e:
        error = str(e)

    query_record = Query(
        dataset_id=dataset.id,
        owner_id=current_user.id,
        question=request.question,
        generated_sql=sql,
        result_json=result,
        chart_config=chart_config,
        chart_image_path=chart_image_path,
        explanation=explanation,
        error=error,
    )
    db.add(query_record)
    db.commit()
    db.refresh(query_record)

    if error:
        raise HTTPException(status_code=400, detail={"sql": sql, "error": error})

    return query_record