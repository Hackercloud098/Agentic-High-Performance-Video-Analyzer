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
    score -= 0.2 * length_diff

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
            score += 1.0
        if token in profile["low_keywords"]:
            score -= 1.0

    # Normalization scores through sigmoid transformation
    normalized_scores = 1/(1+math.exp(-score))

    return normalized_scores

def build_prompt(channel_id: str, summary: str, profile: dict) -> str:
    return (
        f"You are an assistant that suggests compelling YouTube titles.\n"
        f"For channel {channel_id}, high-performing titles are typically around "
        f"{round(profile['avg_title_words'])} words long, often include numbers, "
        "and rarely contain question or exclamation marks.\n"
        f"High-performing words include: {', '.join(profile['top_keywords'][:10])}.\n"
        f"Avoid low-performing words such as: {', '.join(profile['low_keywords'][:10])}.\n"
        f"Given the video summary: \"{summary}\", propose 5 distinct title suggestions.\n"
        "For each title, provide a one-sentence explanation grounded in these data patterns.\n"
        "Respond only with a JSON array of objects (do not use Markdown code blocks), "
        "each having two string keys: \"title\" and \"explanation\"."
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
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:n]
