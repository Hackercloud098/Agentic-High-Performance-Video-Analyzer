import os
from typing import List, Tuple
import re
import openai
import json
from .profiles import _tokenize_no_stop

def score_title(title: str, profile: dict) -> float:
    """Compute a heuristic score for a candidate title based on the channel profile."""
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

    return score

def build_prompt(channel_id: str, summary: str, profile: dict) -> str:
    return f"""
You are an assistant that suggests compelling YouTube titles.
For channel {channel_id}, high-performing titles are typically around {round(profile['avg_title_words'])} words long, often include numbers, and rarely contain question or exclamation marks.
High-performing words include: {', '.join(profile['top_keywords'][:10])}.
Avoid low-performing words such as: {', '.join(profile['low_keywords'][:10])}.
Given the video summary: "{summary}", propose 5 distinct title suggestions.
For each title, provide a one-sentence explanation grounded in these data patterns.
Respond **only** with a JSON array of objects, each having two string keys: "title" and "explanation".  Do not wrap the JSON in markdown or add any commentary.
""".strip()


def call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=350,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]


#def parse_llm_response(response_text: str) -> List[Tuple[str, str]]:
    #"""
    #Parse the LLM’s response to -> (title, explanation) tuples.
    #Assumes the response is formatted as: "1. Title – explanation".
    #"""
    #results = []
    #for line in response_text.split("\n"):
    #    match = re.match(r"\\d+\\.\\s*(.+?)\\s*[–-]\\s*(.+)", line.strip())
    #    if match:
    #        title, explanation = match.groups()
    #        results.append((title.strip(), explanation.strip()))
    #return results

#def parse_llm_response(response_text: str):
#    results = []
#    for line in response_text.split('\n'):
#        line = line.strip()
#        m = re.match(r'^\s*\d+\.\s*(.*)', line)
#        if m:
#            remainder = m.group(1)
#            split = re.split(r'\s*[-–:]\s+', remainder, maxsplit=1)
#            if len(split) == 2:
#                title, explanation = split
#            else:
#                title, explanation = split[0], ""
#            results.append((title.strip(), explanation.strip()))
#    return results

#def generate_titles(channel_id: str, summary: str, profile: dict, n: int = 5) -> List[dict]:
#    """Generate candidate titles, score them, and return the top n with reasoning."""
#    #prompt = build_prompt(channel_id, summary, profile)
#    #raw = call_llm(prompt)
#    #candidates = parse_llm_response(raw)
#    prompt = build_prompt(channel_id, summary, profile)
#    raw = call_llm(prompt)
#    print("RAW LLM OUTPUT:\n", raw)  # debug line
#    candidates = parse_llm_response(raw)
 #   scored = []
 #   for title, explanation in candidates:
 #       score = score_title(title, profile)
 #       scored.append({"title": title, "explanation": explanation, "score": score})
 #   return sorted(scored, key=lambda x: x["score"], reverse=True)[:n]



def parse_llm_response(response_text: str):
    """Assume the LLM response is a pure JSON array; parse it."""
    return json.loads(response_text)

def generate_titles(channel_id: str, summary: str, profile: dict, n: int = 5):
    prompt = build_prompt(channel_id, summary, profile)
    raw = call_llm(prompt)
    candidates = parse_llm_response(raw)  # a list of dicts with 'title' and 'explanation'
    scored = []
    for candidate in candidates:
        score = score_title(candidate['title'], profile)
        scored.append({
            "title": candidate['title'],
            "explanation": candidate['explanation'],
            "score": score,
        })
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:n]
