from typing import List, Tuple, TypedDict
from langgraph.graph import StateGraph, START, END
from .profiles import build_channel_profiles
from .generator import build_prompt, call_llm, parse_llm_response, score_title, normalise_scores



class AgentState(TypedDict, total=False):
    channel_id: str
    summary: str
    profile: dict
    raw_candidates: list
    scored_candidates: list
    final_suggestions: list
    num: int


def generate_candidates_node(state: AgentState) -> AgentState:
    n = int(state.get("num", 5))
    prompt = build_prompt(
        state["channel_id"],
        state["summary"],
        state["profile"],
        num=n,
    )
    raw_response = call_llm(prompt)
    parsed = parse_llm_response(raw_response)
    state["raw_candidates"] = parsed
    return state


def score_candidates_node(state: AgentState) -> AgentState:
    scored = []
    for k in state["raw_candidates"]:
        title = k["title"]
        explanation = k["explanation"]
        raw_score = score_title(title, state["profile"])
        scored.append({
            "title": title,
            "explanation": explanation,
            "score": raw_score,
        })
    state["scored_candidates"] = normalise_scores(scored)
    return state



def select_top_node(state: AgentState) -> AgentState:
    best = sorted(state["scored_candidates"], key=lambda x: x["score"], reverse=True)
    n = state.get("num", 5)
    state["final_suggestions"] = [
        {"title": c["title"], "explanation": c["explanation"]}
        for c in best[:n]
    ]
    return state




def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("generate", generate_candidates_node)
    workflow.add_node("score", score_candidates_node)
    workflow.add_node("select_top", select_top_node)

    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "score")
    workflow.add_edge("score", "select_top")
    workflow.add_edge("select_top", END)

    return workflow.compile()



AGENT_GRAPH = build_agent_graph()
