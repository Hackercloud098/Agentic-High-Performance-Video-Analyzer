from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from .config import settings

from .profiles import build_channel_profiles
from .generator import generate_titles
from .agent_graph import AGENT_GRAPH

app = FastAPI()
PROFILES = None

@app.on_event("startup")
def load_profiles():
    global PROFILES
    PROFILES = build_channel_profiles(settings.training_data_path)

class GenerationRequest(BaseModel):
    channel_id: str
    summary: str


@app.post("/generate_titles")
def generate_titles_graph(req: GenerationRequest, num: int = Query(5, ge = 3, le = 5, description = "Number of titles to return.")):
    profile = PROFILES.get(req.channel_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Unknown channel")
    initial_state = {
        "channel_id": req.channel_id,
        "summary": req.summary,
        "profile": profile,
        "num": num
    }
    final_state = AGENT_GRAPH.invoke(initial_state)
    
    clean_suggestions = [
        {"title": s["title"], "explanation": s["explanation"]}
        for s in final_state["final_suggestions"]
    ]

    return {
    "suggestions": clean_suggestions
    }