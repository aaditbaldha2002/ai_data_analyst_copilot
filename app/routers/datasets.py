import os
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from app.database import get_db
from app.security import get_current_user
from app.models.users import User
from app.models.datasets import Dataset
from app.schemas.datasets import DatasetOut

router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def infer_schema(file_path: str, ext: str) -> dict:
    if ext == ".csv":
        df = pd.read_csv(file_path, nrows=1000)
    else:
        df = pd.read_excel(file_path, nrows=1000)

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "str":
            try:
                df[col] = pd.to_datetime(df[col], errors="raise")
            except (ValueError, TypeError):
                pass

    return {col: str(dtype) for col, dtype in df.dtypes.items()}

@router.post("/upload", response_model=DatasetOut)
def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    try:
        schema = infer_schema(file_path, ext)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    dataset = Dataset(
        owner_id=current_user.id,
        filename=file.filename,
        file_path=file_path,
        schema_json=schema,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return dataset