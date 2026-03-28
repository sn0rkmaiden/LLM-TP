"""
generate_profiles.py
Re-generates LLM text profiles for all users in the movies dataset, then
SBERT-encodes them and saves both the raw text and the embeddings.

This script reproduces the original authors' preprocessing pipeline:
  - For each user, assemble their interaction history as JSON
    {title, description, timestamp} using the recovered item metadata
  - Call the selected LLM with prompt_long / prompt_short / prompt_general
  - SBERT-encode the returned text
  - Save two outputs per profile type:
      *_text.pkl  — DataFrame(user_id, profile_text)   ← needed by generate_contexts.py
      *.pkl       — DataFrame(user_id, profile)         ← 384-dim embedding, drops in as
                                                          replacement for the originals

Usage
-----
python preprocessing/generate_profiles.py --split all
python preprocessing/generate_profiles.py --llm_model microsoft/phi-3-mini-4k-instruct --max_users 50 --split all
python preprocessing/generate_profiles.py --llm_model mistralai/Mistral-7B-Instruct-v0.1 --split all

The model is downloaded from Hugging Face on first run and cached locally.
For gated models (e.g. Llama) set HF_TOKEN in your environment first.

Recommended smaller models:
  - microsoft/phi-3-mini-4k-instruct (3.8B) — default, fastest
  - mistralai/Mistral-7B-Instruct-v0.1 (7B) — balanced quality/speed
  - meta-llama/Llama-2-7b-chat (7B) — good quality, needs HF_TOKEN

Outputs (data/movies/) — filenames include model name
------------------------------------------------------
bert_long_term_user_profiles_{MODEL}_text.pkl
bert_long_term_user_profiles_{MODEL}.pkl
bert_short_term_user_profiles_{MODEL}_text.pkl
bert_short_term_user_profiles_{MODEL}.pkl
bert_general_user_profiles_{MODEL}_text.pkl
bert_general_user_profiles_{MODEL}.pkl
"""

import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATASET             = "movies"
DEFAULT_LLM_MODEL  = "microsoft/phi-3-mini-4k-instruct"  # Changed to smaller model (3.8B)
SBERT_MODEL         = "all-MiniLM-L6-v2"
TEMPERATURE         = 0.2
MAX_NEW_TOKENS      = 256

DATA_DIR      = Path(__file__).resolve().parents[1] / "data" / DATASET
PROMPT_DIR    = Path(__file__).resolve().parents[1] / "prompt"

PROFILE_TYPES = {
    "long":    PROMPT_DIR / "prompt_long.txt",
    "short":   PROMPT_DIR / "prompt_short.txt",
    "general": PROMPT_DIR / "prompt_general.txt",
}

OUT_NAME = {
    "long":    "bert_long_term_user_profiles",
    "short":   "bert_short_term_user_profiles",
    "general": "bert_general_user_profiles",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_history_json(user_id: int, interaction_rows, item_meta: dict) -> str:
    """
    Construct a JSON list of {title, description, timestamp} for a user's history.
    Rows are sorted chronologically (ascending timestamp).
    """
    items = []
    for row in sorted(interaction_rows, key=lambda r: r["timestamp"]):
        iid = int(row["item_id"])
        meta = item_meta.get(iid)
        if meta is None:
            continue
        items.append({
            "title":       meta["title"],
            "description": meta["description_text"],
            "timestamp":   int(row["timestamp"]),
        })
    return json.dumps(items, ensure_ascii=False)


def build_llm_pipeline(model_name: str):
    """Load a HuggingFace text-generation pipeline."""
    print(f"Loading LLM: {model_name} (this may take a while on first run) ...")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = hf_pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=dtype,
    )
    print(f"  LLM loaded on: {pipe.device}")
    return pipe


def llm_call(pipe, prompt_text: str) -> str:
    """Run a single inference through the HF pipeline."""
    messages = [{"role": "user", "content": prompt_text}]
    output = pipe(
        messages,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=True,
    )
    # Most instruction models return the full message list; take the last assistant turn.
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return generated.strip()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main(args):
    llm_model = args.llm_model
    
    # Extract model name for file naming (e.g., "microsoft/phi-3-mini-4k-instruct" → "phi-3-mini-4k-instruct")
    model_name_short = llm_model.split("/")[-1] if "/" in llm_model else llm_model
    print(f"Using model: {llm_model}")
    print(f"Output files will include: {model_name_short}")
    
    # --- Load interactions ---
    splits_to_use = (
        ["train", "validation", "test"] if args.split == "all" else [args.split]
    )
    dfs = []
    for s in splits_to_use:
        df = pd.read_csv(DATA_DIR / f"{s}.csv")
        dfs.append(df)
    interactions = pd.concat(dfs, ignore_index=True)

    # --- Load item metadata (produced by rebuild_metadata.py) ---
    meta_path = DATA_DIR / "item_metadata.pkl"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run preprocessing/rebuild_metadata.py first."
        )
    item_meta_df = load_pickle(str(meta_path))
    item_meta = {
        int(row["item_id"]): {
            "title":            row["title"],
            "description_text": row["description_text"],
        }
        for _, row in item_meta_df.iterrows()
    }
    print(f"Loaded metadata for {len(item_meta)} items.")

    # Group interactions by user
    user_interactions = {}
    for row in interactions.itertuples(index=False):
        uid = int(row.user_id)
        user_interactions.setdefault(uid, []).append(
            {"item_id": int(row.item_id), "timestamp": int(row.timestamp)}
        )
    user_ids = sorted(user_interactions.keys())
    if args.max_users is not None:
        user_ids = user_ids[:args.max_users]
        print(f"Users to process: {len(user_ids)} (capped by --max_users)")
    else:
        print(f"Users to process: {len(user_ids)}")

    # --- Load prompts ---
    prompts = {k: load_prompt(v) for k, v in PROFILE_TYPES.items()}

    # --- Set up LLM pipeline ---
    pipe = build_llm_pipeline(args.llm_model)

    # --- SBERT ---
    print(f"Loading SBERT: {SBERT_MODEL}")
    sbert = SentenceTransformer(SBERT_MODEL)

    # --- Generate profiles ---
    # Storage: profile_type -> {user_id: {"text": str, "emb": np.ndarray}}
    results = {k: {} for k in PROFILE_TYPES}

    for idx, uid in enumerate(user_ids):
        history_json = build_history_json(uid, user_interactions[uid], item_meta)

        if not json.loads(history_json):
            # User has no items with recovered metadata — skip
            continue

        for profile_type, prompt_template in prompts.items():
            full_prompt = prompt_template + "\n\nUser interactions:\n" + history_json
            profile_text = llm_call(pipe, full_prompt)
            results[profile_type][uid] = {"text": profile_text}

        if (idx + 1) % 100 == 0 or (idx + 1) == len(user_ids):
            print(f"  [{idx+1}/{len(user_ids)}] Generated profiles for user {uid}")

    # --- SBERT-encode all profiles ---
    for profile_type in PROFILE_TYPES:
        uid_list   = sorted(results[profile_type].keys())
        texts      = [results[profile_type][u]["text"] for u in uid_list]

        print(f"SBERT-encoding {len(texts)} {profile_type} profiles ...")
        embs = sbert.encode(
            texts,
            batch_size=256,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        # --- Save text pkl (needed by generate_contexts.py) ---
        text_df = pd.DataFrame({
            "user_id":      uid_list,
            "profile_text": texts,
        })
        text_path = DATA_DIR / f"{OUT_NAME[profile_type]}_{model_name_short}_text.pkl"
        save_pickle(text_df, str(text_path))
        print(f"  Saved text profiles → {text_path}")

        # --- Save embedding pkl (drop-in replacement for original) ---
        emb_df = pd.DataFrame({
            "user_id": uid_list,
            "profile": list(embs),
        })
        emb_path = DATA_DIR / f"{OUT_NAME[profile_type]}_{model_name_short}.pkl"
        save_pickle(emb_df, str(emb_path))
        print(f"  Saved embedding profiles → {emb_path}")

    print("Done.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate LLM user profiles for the movies dataset."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Which interaction split(s) to include in user histories.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help="HuggingFace model ID, e.g. 'Qwen/Qwen2.5-7B-Instruct' or "
             "'meta-llama/Llama-3.1-8B-Instruct'.",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=None,
        help="Cap the number of users processed (useful for smoke-tests).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
