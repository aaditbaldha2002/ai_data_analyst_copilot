from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str

    postgres_db_user: str
    postgres_db_password: str
    postgres_db: str

    jwt_secret: str
    jwt_algorithm: str = "HS256"
    openai_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()