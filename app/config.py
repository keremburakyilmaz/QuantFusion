from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/quantfusion"
    REDIS_URL: str = "redis://localhost:6379/0"
    CORS_ORIGINS: str = "http://localhost:3000"
    NVIDIA_API_KEY: str = ""
    NIM_MODEL: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NIM_OCR_MODEL: str = "nvidia/nemotron-ocr-v1"
    DEMO_PORTFOLIO_ID: str = ""
    SECRET_KEY: str = "dev-secret-change-me"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
