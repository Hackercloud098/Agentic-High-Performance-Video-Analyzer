from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .profiles import build_channel_profiles
from .generator import generate_titles

app = FastAPI()
PROFILES = None

@app.on_event("startup")
def load_profiles():
    global PROFILES
    PROFILES = build_channel_profiles("data/electrify__applied_ai_engineer__training_data.csv")

class GenerationRequest(BaseModel):
    channel_id: str
    summary: str

@app.post("/generate_titles")
def generate_titles_endpoint(req: GenerationRequest):
    if PROFILES is None:
        raise HTTPException(status_code=500, detail="Profiles not loaded")
    profile = PROFILES.get(req.channel_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Unknown channel_id")
    suggestions = generate_titles(req.channel_id, req.summary, profile)
    return {"suggestions": suggestions}