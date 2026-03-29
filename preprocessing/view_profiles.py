"""
view_profiles.py
Quick script to view generated user profiles and verify quality.

Usage:
python preprocessing/view_profiles.py --model phi-3-mini-4k-instruct --num_samples 5
python preprocessing/view_profiles.py --model phi-3-mini-4k-instruct --user_id 42
"""

import pickle
import argparse
from pathlib import Path
import pandas as pd


def load_profiles(model_name: str):
    """Load the text profiles from pickle file."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "movies"
    pkl_path = data_dir / f"bert_long_term_user_profiles_{model_name}_text.pkl"
    
    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        print(f"\nAvailable files in {data_dir}:")
        for f in sorted(data_dir.glob("bert_long_term_*_text.pkl")):
            print(f"  - {f.name}")
        exit(1)
    
    df = pd.read_pickle(str(pkl_path))
    print(f"✓ Loaded {len(df)} profiles from {pkl_path.name}\n")
    return df


def show_sample(df, num_samples=5):
    """Display first N user profiles."""
    print(f"{'='*80}")
    print(f"Sample User Profiles (first {num_samples})")
    print(f"{'='*80}\n")
    
    for idx, row in df.head(num_samples).iterrows():
        user_id = row["user_id"]
        profile_text = row["profile_text"]
        
        print(f"User ID: {user_id}")
        print(f"{'-'*80}")
        print(profile_text)
        print(f"\n")


def show_user_profile(df, user_id):
    """Display profile for a specific user ID."""
    row = df[df["user_id"] == user_id]
    
    if row.empty:
        print(f"❌ User {user_id} not found in profiles.")
        print(f"Available user IDs: {sorted(df['user_id'].tolist())}")
        exit(1)
    
    profile_text = row.iloc[0]["profile_text"]
    
    print(f"{'='*80}")
    print(f"Profile for User {user_id}")
    print(f"{'='*80}\n")
    print(profile_text)
    print(f"\n")


def show_stats(df):
    """Show statistics about the profiles."""
    print(f"{'='*80}")
    print(f"Profile Statistics")
    print(f"{'='*80}")
    print(f"Total users: {len(df)}")
    
    df["text_length"] = df["profile_text"].str.len()
    df["text_tokens"] = df["profile_text"].str.split().str.len()
    
    print(f"\nText Length (characters):")
    print(f"  Min: {df['text_length'].min()}")
    print(f"  Max: {df['text_length'].max()}")
    print(f"  Mean: {df['text_length'].mean():.1f}")
    print(f"  Median: {df['text_length'].median():.1f}")
    
    print(f"\nText Length (words):")
    print(f"  Min: {df['text_tokens'].min()}")
    print(f"  Max: {df['text_tokens'].max()}")
    print(f"  Mean: {df['text_tokens'].mean():.1f}")
    print(f"  Median: {df['text_tokens'].median():.1f}")
    
    # Show a few examples
    print(f"\n\nSample profiles (first 3):")
    for idx, row in df.head(3).iterrows():
        print(f"\n  User {row['user_id']} ({row['text_length']} chars, {row['text_tokens']} words):")
        text_preview = row["profile_text"][:150].replace("\n", " ") + "..."
        print(f"  {text_preview}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="View and verify generated user profiles."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="phi-3-mini-4k-instruct",
        help="Model name (e.g. 'phi-3-mini-4k-instruct', 'Mistral-7B-Instruct-v0.1').",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample profiles to display (default: 5).",
    )
    parser.add_argument(
        "--user_id",
        type=int,
        default=None,
        help="Show profile for a specific user ID (overrides --num_samples).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics about all profiles instead of samples.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    df = load_profiles(args.model)
    
    if args.stats:
        show_stats(df)
    elif args.user_id is not None:
        show_user_profile(df, args.user_id)
    else:
        show_sample(df, args.num_samples)
