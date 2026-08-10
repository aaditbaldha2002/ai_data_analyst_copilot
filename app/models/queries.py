from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Text
from sqlalchemy.sql import func
from app.database import Base

class Query(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    question = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=False)
    result_json = Column(JSON, nullable=True)
    chart_config = Column(JSON, nullable=True)
    chart_image_path = Column(String, nullable=True)
    explanation = Column(Text, nullable=True)
    forecast_model_used = Column(String, nullable=True)
    forecast_predictions = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
