from typing import List, Tuple, TypedDict
from langgraph.graph import StateGraph, START, END
from .profiles import build_channel_profiles
from .generator import build_prompt, call_llm, parse_llm_response, score_title



class AgentState(TypedDict, total=False):
    channel_id: str
    summary: str
    profile: dict
    raw_candidates: List[Tuple[str, str]]
    scored_candidates: List[dict]
    final_suggestions: List[dict]
    num: int


def load_profile_node(state: AgentState) -> AgentState:
    return state



def generate_candidates_node(state: AgentState) -> AgentState:
    prompt = build_prompt(state["channel_id"], state["summary"], state["profile"])
    raw_response = call_llm(prompt)
    parsed = parse_llm_response(raw_response)
    state["raw_candidates"] = parsed
    return state



def score_candidates_node(state: AgentState) -> AgentState:
    scored = []
    for candidate in state["raw_candidates"]:
        title = candidate["title"]
        explanation = candidate["explanation"]
        scored.append({
            "title": title,
            "explanation": explanation,
            "score": score_title(title, state["profile"]),
        })
    state["scored_candidates"] = scored
    return state




def select_top_node(state: AgentState) -> AgentState:
    best = sorted(state["scored_candidates"], key=lambda x: x["score"], reverse=True)
    n = state.get("num", 5)
    state["final_suggestions"] = best[:n]
    return state


def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("load_profile", load_profile_node)
    workflow.add_node("generate", generate_candidates_node)
    workflow.add_node("score", score_candidates_node)
    workflow.add_node("select_top", select_top_node)
    workflow.add_edge(START, "load_profile")
    workflow.add_edge("load_profile", "generate")
    workflow.add_edge("generate", "score")
    workflow.add_edge("score", "select_top")
    workflow.add_edge("select_top", END)
    return workflow.compile()

AGENT_GRAPH = build_agent_graph()
