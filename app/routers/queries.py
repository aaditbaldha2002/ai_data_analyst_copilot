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
    try:
        ext = os.path.splitext(dataset.file_path)[1].lower()
        result = run_query(dataset.file_path, ext, sql)
    except Exception as e:
        error = str(e)

    query_record = Query(
        dataset_id=dataset.id,
        owner_id=current_user.id,
        question=request.question,
        generated_sql=sql,
        result_json=result,
        error=error,
    )
    db.add(query_record)
    db.commit()
    db.refresh(query_record)

    if error:
        raise HTTPException(status_code=400, detail={"sql": sql, "error": error})

    return query_record