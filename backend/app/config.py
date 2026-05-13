from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3.5:4B", alias="OLLAMA_MODEL")
    ollama_enable_thinking: bool = Field(default=False, alias="OLLAMA_ENABLE_THINKING")
    app_env: str = Field(default="development", alias="APP_ENV")
    cache_dir: Path = ROOT_DIR / ".cache"

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
