import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.security import get_current_user
from app.models.users import User
from app.models.datasets import Dataset, DatasetColumn
from app.schemas.datasets import DatasetOut
from app.dataset_ingestion import convert_to_parquet, build_dataset_column_rows

router = APIRouter(prefix="/datasets", tags=["datasets"])

TMP_UPLOAD_DIR = "uploaded_files"  # transient landing spot only, pre-conversion
os.makedirs(TMP_UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB — adjust to your needs


@router.post("/upload", response_model=DatasetOut)
def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    # Land the raw upload in a transient tmp path first
    tmp_name = f"{uuid.uuid4().hex}{ext}"
    tmp_path = os.path.join(TMP_UPLOAD_DIR, tmp_name)

    size = 0
    with open(tmp_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                os.remove(tmp_path)
                raise HTTPException(status_code=413, detail="File too large")
            f.write(chunk)

    # Insert a placeholder row first so we have a PK to key the dataset's
    # storage directory off of (convert_to_parquet needs an id up front).
    dataset = Dataset(
        owner_id=current_user.id,
        filename=file.filename,
        raw_file_path=tmp_path,  # temporary; overwritten below
    )
    db.add(dataset)
    db.flush()  # populates dataset.id without committing yet

    try:
        schema = convert_to_parquet(str(dataset.id), tmp_path)
    except Exception as e:
        db.rollback()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")

    # Update the row with real paths + stats now that conversion succeeded
    dataset.raw_file_path = str(_raw_copy_path(dataset.id, ext))
    dataset.parquet_path = str(_parquet_path(dataset.id))
    dataset.row_count = schema.row_count
    dataset.content_hash = schema.content_hash

    db.add_all(build_dataset_column_rows(schema, dataset.id, DatasetColumn))

    db.commit()
    db.refresh(dataset)

    # tmp file was already copied into the dataset's own directory by
    # convert_to_parquet; safe to clean up the transient copy
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return dataset


def _dataset_dir(dataset_id: int):
    from app.dataset_ingestion import DATA_ROOT
    return DATA_ROOT / str(dataset_id)


def _raw_copy_path(dataset_id: int, ext: str):
    return _dataset_dir(dataset_id) / f"raw{ext}"


def _parquet_path(dataset_id: int):
    return _dataset_dir(dataset_id) / "data.parquet"