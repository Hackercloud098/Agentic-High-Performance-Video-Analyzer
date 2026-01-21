import os
from typing import List, Tuple
import re
import openai
import json
from .profiles import _tokenize_no_stop
import math
from .config import settings

def score_title(title: str, profile: dict) -> float:
    """Compute a score for a candidate title based on the channel profile."""
    score = 0.0

    # Length -> channel’s average title length
    words = title.split()
    length_diff = abs(len(words) - profile["avg_title_words_high"])
    # penalise if too short or too long
    score -= 0.5 * length_diff

    # Reward when numbers are present
    if any(char.isdigit() for char in title):
        score += 2.0

    # Penalise question or exclamation marks
    if "?" in title:
        score -= 1.0
    if "!" in title:
        score -= 1.0

    # Reward high-performing keywords and penalise low-performing ones
    tokens = _tokenize_no_stop(title)
    for token in tokens:
        if token in profile["top_keywords"]:
            score += 1.5
        if token in profile["low_keywords"]:
            score -= 0.5
    return score

def normalise_scores(candidates: list[dict]) -> list[dict]:
    """
    Normalizing scores through scaling.
    """
    raw_scores = [c["score"] for c in candidates]
    if not raw_scores:
        return []
    min_score, max_score = min(raw_scores), max(raw_scores)
    diff = max_score - min_score
    denom = diff if diff != 0 else 1
    for c in candidates:
        c["score"] = (c["score"] - min_score) / denom
    return candidates



def build_prompt(channel_id: str, summary: str, profile: dict, num: int = 5) -> str:
    avg_words = round(profile["avg_title_words_high"])
    top_terms = ", ".join(profile.get("top_keywords", [])[:12])
    low_terms = ", ".join(profile.get("low_keywords", [])[:12])

    colon_rate = profile.get("colon_rate_high", 0.0)
    openers = profile.get("top_openers_high", [])[:5]
    openers_str = ", ".join([f"{o['word']} ({o['rate']:.0%})" for o in openers]) or "N/A"

    number_rate = profile.get("number_rate_high", 0.0)
    question_rate = profile.get("question_rate_high", 0.0)
    exclamation_rate = profile.get("exclamation_rate_high", 0.0)

    return (
        "You are generating YouTube titles and one‑sentence, data‑grounded explanations.\n\n"
        f"Channel: {channel_id}\n"
        f"Video summary: \"{summary}\"\n\n"
        f"Return a JSON array of exactly {num} objects. Each object must have keys \"title\" and \"explanation\". "
        "No extra text, no markdown.\n\n"

        "Title guidelines:\n"
        "- Be faithful to the summary: do not introduce new people, organizations, locations, or events not mentioned in the summary.\n"
        "- Preserve key multi‑word phrases from the summary (for example, keep 'solar‑powered tank technology' intact).\n"
        f"- Aim for roughly {avg_words} words per title; variation is OK.\n"
        "- Each title should start with a different first word.\n"
        f"- High‑performing keywords (use sparingly if they fit): {top_terms}\n"
        f"- Low‑performing keywords (avoid unless necessary): {low_terms}\n"
        f"- Colon ':' appears in ~{colon_rate:.0%} of high‑performing titles; numbers in ~{number_rate:.0%}; "
        f"question marks in ~{question_rate:.0%}; exclamation marks in ~{exclamation_rate:.0%}.\n"
        f"- Common opening words: {openers_str}. Use one of these in 1–2 titles if it fits naturally.\n"
        "- Use high‑performing keywords only as adjectives modifying concepts that already appear in the summary. Do not use them as new nouns or subjects; skip any keyword that can’t be used without changing the topic.\n"
        "- Avoid over‑using strong or emotional keywords (e.g. 'fearsome', 'warriors') unless the summary clearly implies conflict or crisis.\n"
        "- Do not force keywords or colons; avoid generic openers like \"Exploring\" or \"A look at\".\n\n"

        "Explanation guidelines:\n"
        "- Exactly one sentence in plain English.\n"
        "- The sentence must follow a strict data‑grounded pattern: only report measurable facts from the title and channel signals. Do not summarize, interpret, or add narrative content.\n"
        f"- Always include the title’s word count and compare it to avg≈{avg_words}.\n"
        "- If the title contains a high‑performing keyword from the list (verbatim, case‑insensitive), mention it as high‑performing keyword \"<kw>\" in the sentence; otherwise omit keywords entirely.\n"
        "- Mention the opening word and its rate only if the first word is one of the common openers listed above.\n"
        "- Mention that the title uses a colon only if a ':' is present; otherwise omit.\n"
        "- Mention numbers only if the title includes digits; otherwise omit numbers.\n"
        "- Never mention the absence of any feature (e.g. never say 'no high‑performing keyword'). Omit clauses for features that aren’t present.\n"
        "- Do not mention '?' or '!'.\n\n"

        "Use this pattern (fill in only the clauses that apply; do not add extra details):\n"
        f"  \"At <WC> words (avg≈{avg_words}), [optional: this title starts with '<FirstWord>' (<Rate>)]"
        " [optional: uses the high‑performing keyword \"<kw>\"]"
        " [optional: contains a colon] [optional: includes the number <N>].\"\n"
    )







def call_llm(prompt: str) -> str:
    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )
    return response["choices"][0]["message"]["content"]


def parse_llm_response(response_text: str):
    """Parse the LLM response."""
    start = response_text.find('[')
    end = response_text.rfind(']')
    if start != -1 and end != -1 and end > start:
        json_str = response_text[start:end+1]
        return json.loads(json_str)
    return []



def generate_titles(channel_id: str, summary: str, profile: dict, n: int = 5):
    prompt = build_prompt(channel_id, summary, profile, num=n)
    raw = call_llm(prompt)
    candidates = parse_llm_response(raw)
    scored = []
    for candidate in candidates:
        score = score_title(candidate['title'], profile)
        scored.append({
            "title": candidate['title'],
            "explanation": candidate['explanation'],
            "score": score,
        })
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)[:n]
    return [{"title": item["title"], "explanation": item["explanation"]} for item in scored]
