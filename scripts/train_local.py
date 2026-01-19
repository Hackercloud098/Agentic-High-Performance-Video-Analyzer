import argparse
from app.profiles import build_channel_profiles, save_channel_profiles
from app.config import settings

def main() -> None:
    parser = argparse.ArgumentParser(description="Build channel profiles from training CSV.")
    parser.add_argument(
        "--input",
        default=settings.training_data_path,
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--output",
        default=settings.channel_profiles_path,
        help="Output path for profiles.",
    )
    args = parser.parse_args()

    profiles = build_channel_profiles(args.input)
    save_channel_profiles(profiles)
    print(f"Saved profiles for {len(profiles)} channels to {args.output}")

if __name__ == "__main__":
    main()

