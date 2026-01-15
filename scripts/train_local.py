"""
Script to build initial channel profiles from the training CSV.
"""

import argparse
from __ import build_channel_profiles, save_channel_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Build channel profiles from training CSV.")
    parser.add_argument("--input", default="data/training.csv", help="Path to training CSV.")
    parser.add_argument("--output", default="artifacts/channel_profiles.json", help="Output path for profiles.")
    args = parser.parse_args()
    profiles = build_channel_profiles(args.input)
    save_channel_profiles(profiles, args.output)
    print(f"Saved profiles for {len(profiles)} channels to {args.output}")


if __name__ == "__main__":
    main()
