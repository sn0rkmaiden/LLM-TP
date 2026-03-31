"""
view_contexts.py
Quick script to inspect generated context embeddings and verify quality.

Usage:
python code/view_contexts.py --num_samples 5
python code/view_contexts.py --user_id 42
python code/view_contexts.py --item_id 100
python code/view_contexts.py --stats
"""

import pickle
import argparse
from pathlib import Path
import numpy as np


def load_contexts(dataset: str = "movies"):
    """Load the context embeddings from pickle file."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / dataset
    pkl_path = data_dir / "bert_context_profiles_test.pkl"
    
    if not pkl_path.exists():
        print(f"❌ File not found: {pkl_path}")
        exit(1)
    
    with open(pkl_path, "rb") as f:
        context_dict = pickle.load(f)
    
    print(f"✓ Loaded context embeddings for {len(context_dict)} (user, item) pairs\n")
    return context_dict


def show_samples(context_dict, num_samples=5):
    """Display first N context entries."""
    print(f"{'='*80}")
    print(f"Sample Context Entries (first {num_samples})")
    print(f"{'='*80}\n")
    
    for idx, (user_id, item_id) in enumerate(sorted(context_dict.keys())[:num_samples]):
        contexts = context_dict[(user_id, item_id)]
        print(f"User {user_id}, Item {item_id}:")
        print(f"  Number of contexts: {len(contexts)}")
        for ctx_idx, emb in enumerate(contexts):
            print(f"    Context {ctx_idx}: shape {emb.shape}, dtype {emb.dtype}")
            print(f"              L2 norm: {np.linalg.norm(emb):.6f}")
        print()


def show_user_contexts(context_dict, user_id):
    """Show all (user, item) pairs for a specific user."""
    user_pairs = [(uid, iid) for uid, iid in context_dict.keys() if uid == user_id]
    
    if not user_pairs:
        print(f"❌ No contexts found for user {user_id}")
        print(f"Available users: {sorted(set(uid for uid, _ in context_dict.keys()))}")
        exit(1)
    
    print(f"{'='*80}")
    print(f"Contexts for User {user_id}")
    print(f"{'='*80}\n")
    print(f"Total items: {len(user_pairs)}\n")
    
    for item_id in sorted([iid for _, iid in user_pairs])[:10]:  # Show first 10
        contexts = context_dict[(user_id, item_id)]
        print(f"  Item {item_id}: {len(contexts)} contexts, dims {contexts[0].shape}")
    
    if len(user_pairs) > 10:
        print(f"  ... and {len(user_pairs) - 10} more items")


def show_item_contexts(context_dict, item_id):
    """Show all (user, item) pairs for a specific item."""
    item_pairs = [(uid, iid) for uid, iid in context_dict.keys() if iid == item_id]
    
    if not item_pairs:
        print(f"❌ No contexts found for item {item_id}")
        print(f"Available items: {sorted(set(iid for _, iid in context_dict.keys()))}")
        exit(1)
    
    print(f"{'='*80}")
    print(f"Contexts for Item {item_id}")
    print(f"{'='*80}\n")
    print(f"Total users: {len(item_pairs)}\n")
    
    for user_id in sorted([uid for uid, _ in item_pairs])[:10]:  # Show first 10
        contexts = context_dict[(user_id, item_id)]
        print(f"  User {user_id}: {len(contexts)} contexts, dims {contexts[0].shape}")
    
    if len(item_pairs) > 10:
        print(f"  ... and {len(item_pairs) - 10} more users")


def show_stats(context_dict):
    """Show statistics about the generated contexts."""
    print(f"{'='*80}")
    print(f"Context Statistics")
    print(f"{'='*80}\n")
    
    total_pairs = len(context_dict)
    total_contexts = sum(len(contexts) for contexts in context_dict.values())
    
    print(f"Total (user, item) pairs: {total_pairs}")
    print(f"Total contexts: {total_contexts}")
    
    # Get context counts per pair
    ctx_counts = [len(contexts) for contexts in context_dict.values()]
    print(f"\nContexts per pair:")
    print(f"  Min: {min(ctx_counts)}")
    print(f"  Max: {max(ctx_counts)}")
    print(f"  Mean: {np.mean(ctx_counts):.2f}")
    print(f"  Median: {np.median(ctx_counts):.1f}")
    print(f"  Unique counts: {sorted(set(ctx_counts))}")
    
    # Get embedding dimensions
    sample_emb = next(iter(context_dict.values()))[0]
    print(f"\nEmbedding dimensions:")
    print(f"  Shape: {sample_emb.shape}")
    print(f"  Dtype: {sample_emb.dtype}")
    
    # Check if embeddings are normalized (L2 norm ~= 1)
    sample_norms = [np.linalg.norm(emb) for embs in list(context_dict.values())[:100] for emb in embs]
    print(f"  L2 norm (first 100 contexts):")
    print(f"    Min: {min(sample_norms):.6f}")
    print(f"    Max: {max(sample_norms):.6f}")
    print(f"    Mean: {np.mean(sample_norms):.6f}")
    are_normalized = all(0.99 < norm < 1.01 for norm in sample_norms)
    print(f"    Normalized (L2≈1): {are_normalized}")
    
    # User and item coverage
    unique_users = len(set(uid for uid, _ in context_dict.keys()))
    unique_items = len(set(iid for _, iid in context_dict.keys()))
    print(f"\nCoverage:")
    print(f"  Unique users: {unique_users}")
    print(f"  Unique items: {unique_items}")
    print(f"  Sparsity: {total_pairs / (unique_users * unique_items) * 100:.2f}%")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inspect generated context embeddings."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="movies",
        help="Dataset name (default: movies).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample (user, item) pairs to display (default: 5).",
    )
    parser.add_argument(
        "--user_id",
        type=int,
        default=None,
        help="Show all contexts for a specific user ID.",
    )
    parser.add_argument(
        "--item_id",
        type=int,
        default=None,
        help="Show all contexts for a specific item ID.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show comprehensive statistics about all contexts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    context_dict = load_contexts(args.dataset)
    
    if args.stats:
        show_stats(context_dict)
    elif args.user_id is not None:
        show_user_contexts(context_dict, args.user_id)
    elif args.item_id is not None:
        show_item_contexts(context_dict, args.item_id)
    else:
        show_samples(context_dict, args.num_samples)
