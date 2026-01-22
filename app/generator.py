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

    high_kws = profile.get("top_keywords", [])[:12]
    low_kws  = profile.get("low_keywords", [])[:12]
    high_kw_str = ", ".join(high_kws) or "N/A"
    low_kw_str  = ", ".join(low_kws)  or "N/A"

    openers = profile.get("top_openers_high", [])[:5]
    opener_map = ", ".join([f"{o['word']}={o['rate']:.0%}" for o in openers]) or "N/A"
    opener_words = [o["word"] for o in openers]
    opener_words_str = ", ".join(opener_words) or "N/A"

    colon_rate = profile.get("colon_rate_high", 0.0)
    number_rate = profile.get("number_rate_high", 0.0)
    question_rate = profile.get("question_rate_high", 0.0)
    exclamation_rate = profile.get("exclamation_rate_high", 0.0)

    return (
        "Generate high-quality YouTube titles and 1-sentence explanations.\n"
        "Output ONLY valid JSON: an array of exactly "
        f"{num} objects with keys \"title\" and \"explanation\".\n\n"
        f"Channel: {channel_id}\n"
        f"Summary: {summary}\n\n"

        "Style signals (high performers):\n"
        f"- avg words ~{avg_words}\n"
        f"- openers (word=rate): {opener_map}\n"
        f"- HIGH keywords (single words): {high_kw_str}\n"
        f"- avoid keywords: {low_kw_str}\n"
        f"- rates: ':' {colon_rate:.0%}, numbers {number_rate:.0%}, '?' {question_rate:.0%}, '!' {exclamation_rate:.0%}\n\n"

        "TITLE RULES:\n"
        "- Faithful to the summary; do NOT add entities/places/events not mentioned.\n"
        f"- Aim for {avg_words} words (+/-2).\n"
        "- Each title starts with a DIFFERENT first word.\n"
        "- Use 0-2 HIGH keywords total across all titles, and never repeat the same HIGH keyword.\n"
        "- Avoid low keywords unless necessary.\n"
        f"- If ':' rate is ~0%, do NOT use ':'. If '!' rate is ~0%, do NOT use '!'. If '?' rate is ~0%, do NOT use '?'.\n\n"

        "EXPLANATION RULES (STRICT, NO STORY):\n"
        "- Exactly ONE sentence.\n"
        f"- MUST begin: \"At <WC> words (avg≈{avg_words})\".\n"
        "- Then add ONLY these optional clauses, ONLY if true:\n"
        f"  * \"; opener '<FirstWord>' (<Rate>)\" ONLY if FirstWord is exactly one of [{opener_words_str}] (case-insensitive), and Rate must be copied from the opener map.\n"
        f"  * \"; high keyword '<kw>'\" ONLY if <kw> is exactly one of [{high_kw_str}] (case-insensitive) AND appears as a whole word in the title.\n"
        "  * \"; number <N>\" ONLY if digits appear in the title.\n"
        "  * \"; colon\" ONLY if ':' appears in the title.\n"
        "- Do NOT add any other text (no mission/impact/meaning/scientists/etc).\n"
        "- NEVER mention absence (no 'no', 'none', 'does not'). If not present, omit the clause.\n"
        "- Self-check before output: if you wrote an opener/keyword/number/colon clause and it is not provably true from the title + lists, DELETE that clause.\n"
       '- The "number <N>" clause is allowed ONLY if the title text contains a digit 0-9 (e.g., "3", "10"). Never use the word count as the number.\n'
        "- If there is no digit in the title, do not mention numbers at all.\n"
        "- If you include an opener rate, write it as a percentage like 37%.\n"
        "- Word count = split on spaces.\n"
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
