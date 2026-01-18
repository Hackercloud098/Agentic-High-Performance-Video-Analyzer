from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    model_name: str = "gpt-4o-mini"

    default_num_titles: int = 5
    default_temperature: float = 0.7

    channel_profiles_path: str = "artifacts/channel_profiles.json"


settings = Settings()
