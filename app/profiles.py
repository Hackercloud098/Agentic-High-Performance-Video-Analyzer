import os
import json
from collections import Counter
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from app.config import settings
from typing import Optional

def _tokenize_no_stop(text: str) -> list[str]:
    """
    Tokenizes a string to lower‑case words, removing punctuation, stop words,
    single-character tokens, and tokens containing only digits.
    """
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return [
        w
        for w in tokens
        if (w not in ENGLISH_STOP_WORDS)
        and len(w) >= 2
        and not(w.isdigit())
    ]

def get_top_openers(titles: pd.Series, k: int = 5):
    first_words = (
        titles.astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .dropna()
        .str.lower()
    )

    if first_words.empty:
        return []

    counts = first_words.value_counts()
    total = int(counts.sum())
    top = counts.head(k)

    return [{"word": w, "rate": float(c) / total} for w, c in top.items()]

def build_channel_profiles(input_path: str) -> dict[str, dict]:
    df = pd.read_csv(input_path)
    profiles: dict[str, dict] = {}
    df["clean_title"] = df["title"].apply(lambda t: " ".join(_tokenize_no_stop(t)))

    for channel_id, group in df.groupby("channel_id"):
        profile: dict = {}

        # Split into high/low performance videos
        threshold = group["views_in_period"].quantile(0.75)
        high_group = group[group["views_in_period"] >= threshold]
        low_group = group[group["views_in_period"] < threshold]

        # Basic stats
        profile["num_videos"] = len(group)
        profile["mean_views"] = group["views_in_period"].mean()
        profile["median_views"] = group["views_in_period"].median()
        profile["avg_title_words_high"] = high_group["title"].str.split().apply(len).mean()
        profile["avg_title_chars_high"] = high_group["title"].str.len().mean()
        profile["number_rate_high"] = high_group["title"].str.contains(r"\d").mean()
        profile["question_rate_high"] = high_group["title"].str.contains(r"\?").mean()
        profile["exclamation_rate_high"] = high_group["title"].str.contains(r"!").mean()
        profile["colon_rate_high"] = high_group["title"].astype(str).str.contains(r":").mean()
        if len(high_group) >= 3:
            profile["top_openers_high"] = get_top_openers(high_group["title"], k=5)
        else:
            profile["top_openers_high"] = get_top_openers(group["title"], k=5)
        
        # Fit TF‑IDF on all titles for a given channel
        vectoriser = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )
        titles_all = pd.concat([high_group["clean_title"], low_group["clean_title"]])
        vectoriser.fit(titles_all)

        tfidf_high = vectoriser.transform(high_group["clean_title"])
        tfidf_low  = vectoriser.transform(low_group["clean_title"])
        vocab = vectoriser.get_feature_names_out()

        # Compute average TF‑IDF score per token in each group
        high_avg = tfidf_high.sum(axis=0).A1 / max(len(high_group), 1)
        low_avg  = tfidf_low.sum(axis=0).A1  / max(len(low_group), 1)

        # Compute score ratio
        tfidf_scores = {
            token: (high_avg[idx] + 1e-6) / (low_avg[idx] + 1e-6)
            for idx, token in enumerate(vocab)
        }
        sorted_tfidf = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        profile["top_keywords"] = [t for t, _ in sorted_tfidf[:20]]
        profile["low_keywords"] = [t for t, _ in sorted_tfidf[-20:]]


        profiles[channel_id] = profile

    return profiles


def save_channel_profiles(profiles: dict, output_path: Optional[str] = None) -> None:
    path = output_path or settings.channel_profiles_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(profiles, f, indent=2)
