from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseSettings):
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
    deepseek_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="qwen3.5:4B", alias="OLLAMA_MODEL")
    app_env: str = Field(default="development", alias="APP_ENV")
    cache_dir: Path = ROOT_DIR / ".cache"
    judge_api_key: str | None = Field(default=None, alias="JUDGE_API_KEY")
    judge_model: str = Field(default="deepseek-chat", alias="JUDGE_MODEL")
    judge_base_url: str = Field(default="https://api.deepseek.com", alias="JUDGE_BASE_URL")

    model_config = SettingsConfigDict(extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
