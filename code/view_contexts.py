"""
view_contexts.py

Inspect generated context texts (and embeddings if available) for adequacy.

Usage
-----
python code/view_contexts.py --num_samples 5
python code/view_contexts.py --user_id 42
python code/view_contexts.py --item_id 100
python code/view_contexts.py --stats
python code/view_contexts.py --llm_model microsoft/phi-3-mini-4k-instruct --num_samples 3
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_profile_text_file(data_dir: Path, llm_model: str, explicit_profile_file: str | None):
    if explicit_profile_file:
        p = Path(explicit_profile_file)
        if not p.is_absolute() and "/" not in str(p) and "\\" not in str(p):
            p = data_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Profile text file not found: {p}")
        return p

    model_name_short = llm_model.split("/")[-1] if "/" in llm_model else llm_model
    candidates = list(data_dir.glob(f"bert_long_term_user_profiles_{model_name_short}_N*_T*_text.pkl"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    fallback = data_dir / f"bert_long_term_user_profiles_{model_name_short}_text.pkl"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"No matching profile text file found for model {llm_model} in {data_dir}"
    )


def load_artifacts(dataset="movies", llm_model="microsoft/phi-3-mini-4k-instruct", profile_file=None):
    data_dir = Path(__file__).resolve().parents[1] / "data" / dataset

    ctx_text_path = data_dir / "bert_context_profiles_test_text.pkl"
    ctx_emb_path = data_dir / "bert_context_profiles_test.pkl"
    item_meta_path = data_dir / "item_metadata.pkl"
    profile_text_path = find_profile_text_file(data_dir, llm_model, profile_file)

    if not ctx_text_path.exists():
        raise FileNotFoundError(
            f"Missing raw context text file: {ctx_text_path}\n"
            f"Use the patched generate_contexts.py that saves bert_context_profiles_test_text.pkl"
        )
    if not item_meta_path.exists():
        raise FileNotFoundError(f"Missing item metadata file: {item_meta_path}")

    ctx_text_dict = load_pickle(ctx_text_path)
    ctx_emb_dict = load_pickle(ctx_emb_path) if ctx_emb_path.exists() else None
    item_meta_df = load_pickle(item_meta_path)
    profile_text_df = load_pickle(profile_text_path)

    item_meta = {
        int(row["item_id"]): {
            "title": str(row["title"]),
            "description_text": str(row["description_text"]),
        }
        for _, row in item_meta_df.iterrows()
    }

    user_profiles = {
        int(row["user_id"]): str(row["profile_text"])
        for _, row in profile_text_df.iterrows()
    }

    print(f"✓ Loaded raw contexts for {len(ctx_text_dict)} (user, item) pairs")
    if ctx_emb_dict is not None:
        print(f"✓ Loaded context embeddings for {len(ctx_emb_dict)} (user, item) pairs")
    else:
        print("• No embedding file found; text-only inspection mode")
    print(f"✓ Loaded item metadata for {len(item_meta)} items")
    print(f"✓ Loaded long-term profiles for {len(user_profiles)} users\n")

    return ctx_text_dict, ctx_emb_dict, item_meta, user_profiles


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_set(s: str):
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", s.lower()))


def assess_context(context: str, profile: str, item_title: str, item_description: str):
    """
    Heuristic checks for adequacy. These are not perfect, but useful triage.
    """
    issues = []
    notes = []

    ctx = context.strip()
    ctx_norm = normalize_text(ctx)
    title_norm = normalize_text(item_title)
    profile_tokens = token_set(profile)
    item_tokens = token_set(item_description)
    ctx_tokens = token_set(ctx)

    # Length checks: prompt asks for roughly 200-400 characters
    n_chars = len(ctx)
    if n_chars < 120:
        issues.append(f"too_short({n_chars} chars)")
    elif n_chars > 500:
        issues.append(f"too_long({n_chars} chars)")
    else:
        notes.append(f"length_ok({n_chars} chars)")

    # Title leakage
    if title_norm and title_norm in ctx_norm:
        issues.append("mentions_title")

    # Profile restatement heuristic
    if len(ctx_tokens) > 0:
        overlap_profile = len(ctx_tokens & profile_tokens) / max(1, len(ctx_tokens))
        if overlap_profile > 0.55:
            issues.append(f"profile_restatement_like(overlap={overlap_profile:.2f})")
        else:
            notes.append(f"profile_overlap={overlap_profile:.2f}")

    # Plot-retelling heuristic
    if len(ctx_tokens) > 0:
        overlap_item = len(ctx_tokens & item_tokens) / max(1, len(ctx_tokens))
        if overlap_item > 0.55:
            issues.append(f"plot_retelling_like(overlap={overlap_item:.2f})")
        else:
            notes.append(f"item_overlap={overlap_item:.2f}")

    # Style hints: context should sound situational / temporary
    situational_cues = [
        "tonight", "today", "right now", "after", "feeling", "mood", "wanting",
        "looking for", "in the mood", "winding down", "weekend", "evening",
        "late-night", "reflective", "comfort", "escap", "unwind", "something"
    ]
    if any(cue in ctx_norm for cue in situational_cues):
        notes.append("situational_language_present")
    else:
        issues.append("weak_situational_signal")

    return issues, notes


def print_pair(user_id, item_id, ctx_texts, ctx_emb_dict, item_meta, user_profiles, show_profile=True, max_desc_chars=500):
    title = item_meta.get(item_id, {}).get("title", "<missing title>")
    description = item_meta.get(item_id, {}).get("description_text", "<missing description>")
    profile = user_profiles.get(user_id, "<missing profile>")

    print("=" * 100)
    print(f"User {user_id} | Item {item_id}")
    print(f"Title: {title}")
    print("-" * 100)

    if show_profile:
        print("LONG-TERM USER PROFILE:")
        print(profile)
        print()

    print("ITEM DESCRIPTION:")
    print(description[:max_desc_chars] + ("..." if len(description) > max_desc_chars else ""))
    print()

    if ctx_emb_dict is not None and (user_id, item_id) in ctx_emb_dict:
        embs = ctx_emb_dict[(user_id, item_id)]
        print(f"Embedding summary: {len(embs)} contexts")
        for i, emb in enumerate(embs):
            print(f"  Context {i+1}: shape={emb.shape}, dtype={emb.dtype}, L2={np.linalg.norm(emb):.6f}")
        print()

    print("GENERATED CONTEXTS:")
    for i, ctx in enumerate(ctx_texts, start=1):
        issues, notes = assess_context(ctx, profile, title, description)
        print(f"\nContext {i}:")
        print(ctx)
        print(f"  Issues: {issues if issues else 'none'}")
        print(f"  Notes:  {notes if notes else 'none'}")
    print()


def show_samples(ctx_text_dict, ctx_emb_dict, item_meta, user_profiles, num_samples=5, show_profile=True):
    keys = sorted(ctx_text_dict.keys())[:num_samples]
    print(f"{'='*100}")
    print(f"Sample Context Entries (first {len(keys)})")
    print(f"{'='*100}\n")
    for user_id, item_id in keys:
        print_pair(user_id, item_id, ctx_text_dict[(user_id, item_id)], ctx_emb_dict, item_meta, user_profiles, show_profile=show_profile)


def show_user_contexts(ctx_text_dict, ctx_emb_dict, item_meta, user_profiles, user_id, show_profile=True):
    keys = sorted([(u, i) for (u, i) in ctx_text_dict.keys() if u == user_id], key=lambda x: x[1])
    if not keys:
        print(f"❌ No contexts found for user {user_id}")
        return
    print(f"{'='*100}")
    print(f"Contexts for User {user_id} ({len(keys)} items)")
    print(f"{'='*100}\n")
    for u, i in keys:
        print_pair(u, i, ctx_text_dict[(u, i)], ctx_emb_dict, item_meta, user_profiles, show_profile=show_profile)


def show_item_contexts(ctx_text_dict, ctx_emb_dict, item_meta, user_profiles, item_id, show_profile=True, limit_users=10):
    keys = sorted([(u, i) for (u, i) in ctx_text_dict.keys() if i == item_id], key=lambda x: x[0])
    if not keys:
        print(f"❌ No contexts found for item {item_id}")
        return
    print(f"{'='*100}")
    print(f"Contexts for Item {item_id} ({len(keys)} users)")
    print(f"{'='*100}\n")
    for u, i in keys[:limit_users]:
        print_pair(u, i, ctx_text_dict[(u, i)], ctx_emb_dict, item_meta, user_profiles, show_profile=show_profile)
    if len(keys) > limit_users:
        print(f"... and {len(keys) - limit_users} more users")


def show_stats(ctx_text_dict, ctx_emb_dict):
    print(f"{'='*100}")
    print("Context Statistics")
    print(f"{'='*100}\n")

    total_pairs = len(ctx_text_dict)
    total_contexts = sum(len(v) for v in ctx_text_dict.values())
    unique_users = len(set(u for u, _ in ctx_text_dict.keys()))
    unique_items = len(set(i for _, i in ctx_text_dict.keys()))
    ctx_counts = [len(v) for v in ctx_text_dict.values()]
    lengths = [len(t) for contexts in ctx_text_dict.values() for t in contexts]

    print(f"Total (user, item) pairs: {total_pairs}")
    print(f"Total contexts: {total_contexts}")
    print(f"Unique users: {unique_users}")
    print(f"Unique items: {unique_items}")
    print()
    print("Contexts per pair:")
    print(f"  Min: {min(ctx_counts)}")
    print(f"  Max: {max(ctx_counts)}")
    print(f"  Mean: {np.mean(ctx_counts):.2f}")
    print(f"  Median: {np.median(ctx_counts):.2f}")
    print(f"  Unique counts: {sorted(set(ctx_counts))}")
    print()
    print("Context text length (characters):")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")

    if ctx_emb_dict is not None and len(ctx_emb_dict) > 0:
        sample_emb = next(iter(ctx_emb_dict.values()))[0]
        sample_norms = [np.linalg.norm(emb) for embs in list(ctx_emb_dict.values())[:100] for emb in embs]
        print()
        print("Embedding info:")
        print(f"  Shape: {sample_emb.shape}")
        print(f"  Dtype: {sample_emb.dtype}")
        print(f"  L2 norm min/mean/max (sample): {min(sample_norms):.6f} / {np.mean(sample_norms):.6f} / {max(sample_norms):.6f}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inspect generated context texts and embeddings.")
    parser.add_argument("--dataset", type=str, default="movies")
    parser.add_argument("--llm_model", type=str, default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--profile_file", type=str, default=None,
                        help="Optional explicit long-profile text file to load.")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--user_id", type=int, default=None)
    parser.add_argument("--item_id", type=int, default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--hide_profile", action="store_true",
                        help="Hide the long-term profile text in the output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    ctx_text_dict, ctx_emb_dict, item_meta, user_profiles = load_artifacts(
        dataset=args.dataset,
        llm_model=args.llm_model,
        profile_file=args.profile_file,
    )

    show_profile = not args.hide_profile

    if args.stats:
        show_stats(ctx_text_dict, ctx_emb_dict)
    elif args.user_id is not None:
        show_user_contexts(ctx_text_dict, ctx_emb_dict, item_meta, user_profiles, args.user_id, show_profile=show_profile)
    elif args.item_id is not None:
        show_item_contexts(ctx_text_dict, ctx_emb_dict, item_meta, user_profiles, args.item_id, show_profile=show_profile)
    else:
        show_samples(ctx_text_dict, ctx_emb_dict, item_meta, user_profiles, num_samples=args.num_samples, show_profile=show_profile)