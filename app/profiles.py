import os
import json
from collections import Counter
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def _tokenize_no_stop(text: str) -> list[str]:
    """Tokenizes a string to lower‑case words, removing punctuation and stop words."""
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    return [w for w in tokens if w not in ENGLISH_STOP_WORDS]

def build_channel_profiles(input_path: str) -> dict[str, dict]:
    """Load CSV and build a profile for each channel, where each profile contains aggregate statistics and lists of high/low‑performing keywords."""
    df = pd.read_csv(input_path)
    profiles = {}
    threshold = df["views_in_period"].quantile(0.75)

    for channel_id, group in df.groupby("channel_id"):
        profile = {}
        
        # stats on views, linguistic features, word and char counts
        profile["num_videos"] = len(group)
        profile["mean_views"] = group["views_in_period"].mean()
        profile["median_views"] = group["views_in_period"].median()
        profile["avg_title_words"] = group["title"].str.split().apply(len).mean()
        profile["avg_title_chars"] = group["title"].str.len().mean()
        profile["number_rate"] = group["title"].str.contains(r"\d").mean()
        profile["question_rate"] = group["title"].str.contains(r"\?").mean()
        profile["exclamation_rate"] = group["title"].str.contains(r"!").mean()

        # high and low performing keywords
        high_perf = group[group["views_in_period"] >= threshold]
        low_perf = group[group["views_in_period"] < threshold]

        high_words = Counter()
        for title in high_perf["title"]:
            high_words.update(_tokenize_no_stop(title))
        
        low_words = Counter()
        for title in low_perf["title"]:
            low_words.update(_tokenize_no_stop(title))

        # top 20 keywords per category
        profile["top_keywords"] = [w for w, _ in high_words.most_common(20)]
        profile["low_keywords"] = [w for w, _ in low_words.most_common(20)]


        profiles[channel_id] = profile

    return profiles

def save_channel_profiles(profiles: dict, output_path: str) -> None:
    """Save the profiles dictionary to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profiles, f, indent=2)
