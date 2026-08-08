from fastapi import FastAPI
from app.routers import auth, datasets, queries

app = FastAPI(title="AI Data Analyst Copilot")

app.include_router(auth.router)
app.include_router(datasets.router)
app.include_router(queries.router)

@app.get("/health")
def health():
    return {"status": "ok"}