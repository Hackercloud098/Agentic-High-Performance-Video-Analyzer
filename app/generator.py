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
    length_diff = abs(len(words) - profile["avg_title_words"])
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
    avg_words = round(profile["avg_title_words"])
    top_terms = ", ".join(profile["top_keywords"][:12])
    low_terms = ", ".join(profile["low_keywords"][:12])

    number_rate = profile.get("number_rate", 0.0)
    question_rate = profile.get("question_rate", 0.0)
    exclamation_rate = profile.get("exclamation_rate", 0.0)

    return (
        "You write YouTube titles and data-grounded, short narrative explanations.\n\n"
        f"Channel: {channel_id}\n"
        f"Video summary: \"{summary}\"\n\n"
        f"Return ONLY valid JSON: an array of exactly {num} objects.\n"
        "Each object MUST have keys: \"title\" and \"explanation\". No extra text, no markdown.\n"
        "Never return an empty array.\n\n"

        "TITLE RULES:\n"
        "- Faithful to the summary (no new claims).\n"
        f"- Aim for around {avg_words} words (not all titles must be identical length).\n"
        "- Each title starts with a different first word.\n"
        "- IMPORTANT: try to include high-performing keywords from the list below, verbatim (case-insensitive match).\n"
        "- Titles should try using DIFFERENT high-performing keywords if possible.\n"
        "- In the other titles, do not force keywords.\n"
        "- Avoid generic openers like \"Exploring\" or \"A look at\" unless necessary.\n\n"


        "CHANNEL SIGNALS:\n"
        f"- Typical length ~{avg_words} words.\n"
        f"- Numbers appear ~{number_rate:.0%}; '?' ~{question_rate:.0%}; '!' ~{exclamation_rate:.0%}.\n"
        f"- High-performing keywords list: {top_terms}\n"
        f"- Low-performing keywords list: {low_terms}\n\n"

        "EXPLANATION RULES (must be grounded):\n"
        "- Exactly ONE sentence, natural English.\n"
        "- ONLY use facts you can verify from (a) the title text and (b) the lists/stats above.\n"
        "- Compute word count as: split the title on spaces (hyphenated counts as 1).\n"
        "- The sentence MUST include the word count and 'avg≈{avg_words}'.\n"
        "- If the title contains any number (digits), mention that number; otherwise omit numbers.\n"
        "- If any high-performing keywords appear verbatim (case-insensitive) in the title, mention them as: high-performing keyword \"<kw>\".\n"
        "- If no high-performing keywords appear in the title, omit any mention of high keywords entirely.\n"
        "- If any low-performing keywords appear verbatim (case-insensitive) in the title, mention only those; otherwise OMIT low keywords entirely.\n"
        "- Do NOT mention punctuation at all.\n"
        "- Do NOT say 'none' / 'no high keywords' / 'no low keywords' — just omit those parts.\n\n"

       "Recommended sentence pattern (flexible):\n"
        f"\"At <WC> words (avg≈{avg_words}), this title is <close/shorter/longer> to the channel's norm"
        " and uses the high-performing keyword \"<kw>\"\""
        " and includes the number <N>.\"\n"
        "Notes: include the keyword clause only if a high keyword is actually in the title; include the number clause only if a digit is in the title.\n"

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
    prompt = build_prompt(channel_id, summary, profile)
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
