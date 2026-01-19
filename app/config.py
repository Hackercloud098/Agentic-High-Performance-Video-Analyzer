import os

training_data_path = os.environ.get(
    "DATA_PATH",
    "data/electrify__applied_ai_engineer__training_data.csv",
)

channel_profiles_path = os.environ.get(
    "PROFILES_OUTPUT_PATH",
    "artifacts/channel_profiles.json",
)

class Settings:
    """
    Settings container for config variables.
    """
    def __init__(self):
        self.training_data_path: str = training_data_path
        self.openai_api_key: str | None = os.environ.get("OPENAI_API_KEY")
        self.model_name: str = "gpt-4o-mini"
        self.temperature: float = 0.7
        self.channel_profiles_path: str = channel_profiles_path
        self.max_tokens: int = 350

settings = Settings()
