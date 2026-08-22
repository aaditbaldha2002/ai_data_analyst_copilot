from dataclasses import dataclass
from sqlalchemy.orm import Session


@dataclass
class GraphContext:
    db: Session