"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    secret_key: str = "change-me-to-a-random-secret-key"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Database
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "areteusml"
    postgres_password: str = "areteusml_dev"
    postgres_db: str = "areteusml"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "areteusml"

    # Model
    model_name: str = "answerdotai/ModernBERT-base"
    model_path: str = "ml/models/production"
    onnx_model_path: str = "ml/models/onnx/int8"

    # Monitoring
    evidently_workspace: str = "monitoring/workspace"
    drift_check_interval: int = 300

    # Argilla
    argilla_api_url: str = "http://localhost:6900"
    argilla_api_key: str = "argilla.apikey"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"


settings = Settings()
