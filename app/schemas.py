from pydantic import BaseModel, Field


class GenerateTitlesRequest(BaseModel):
    channel_id: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    style: str | None = None


class GenerateTitlesResponse(BaseModel):
    titles: list[str]
    debug: dict | None = None
