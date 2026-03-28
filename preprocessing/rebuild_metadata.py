"""
rebuild_metadata.py
Recovers the item_id → (title, description_text) mapping by streaming the
Amazon Reviews 2023 Movies & TV metadata from Hugging Face and matching each
item's SBERT embedding against the pre-computed bert_item_features.pkl.

How the match works
-------------------
The original authors encoded item descriptions with SBERT (all-MiniLM-L6-v2)
and stored the resulting L2-normalised 384-dim vectors.  SBERT is deterministic,
so encoding the same text again produces the same vector.  A cosine similarity
(= dot product for L2-normalised vectors) >= MATCH_THRESHOLD identifies a hit.

Output
------
data/movies/item_metadata.pkl  — DataFrame with columns:
    item_id (int)  |  asin (str)  |  title (str)  |  description_text (str)

Usage
-----
pip install datasets sentence-transformers
python preprocessing/rebuild_metadata.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATASET_REPO   = "McAuley-Lab/Amazon-Reviews-2023"
METADATA_NAME  = "raw_meta_Movies_and_TV"
SBERT_MODEL    = "all-MiniLM-L6-v2"
MIN_DESC_LEN   = 500          # characters — same filter the authors used
MATCH_THRESHOLD = 0.9999      # cosine similarity threshold for exact match
BATCH_SIZE     = 256          # SBERT encoding batch size

DATA_DIR  = Path(__file__).resolve().parents[1] / "data" / "movies"
ITEM_PKL  = DATA_DIR / "bert_item_features.pkl"
OUT_PKL   = DATA_DIR / "item_metadata.pkl"


# ------------------------------------------------------------------
# Load known embeddings
# ------------------------------------------------------------------
def load_known_embeddings(pkl_path: Path):
    df = pickle.load(open(pkl_path, "rb"))
    item_ids   = df["item_id"].tolist()
    embeddings = np.stack(df["description"].values).astype(np.float32)
    # Ensure L2-normalised (they already are, but be safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1, norms)
    return item_ids, embeddings


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print(f"Loading known embeddings from {ITEM_PKL} ...")
    item_ids, known_emb = load_known_embeddings(ITEM_PKL)
    n_target = len(item_ids)
    print(f"  {n_target} items to recover.")

    # Map from matrix row index → item_id for matched items
    found = {}          # item_id -> {"asin": str, "title": str, "description_text": str}
    found_mask = np.zeros(n_target, dtype=bool)   # True once item i is matched

    sbert = SentenceTransformer(SBERT_MODEL)

    print(f"Streaming {METADATA_NAME} from Hugging Face ...")
    dataset = load_dataset(
        DATASET_REPO,
        METADATA_NAME,
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    # Buffer items whose description passes the length filter
    buffer_asins  = []
    buffer_titles = []
    buffer_descs  = []

    def flush_buffer():
        """Encode the current buffer, match against known embeddings, record hits."""
        if not buffer_descs:
            return

        vecs = sbert.encode(
            buffer_descs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)   # (B, 384)

        # Cosine similarity matrix: (B, n_target)
        sims = vecs @ known_emb.T    # dot product of L2-normalised vectors

        for b_idx in range(len(buffer_descs)):
            best_idx = int(np.argmax(sims[b_idx]))
            if sims[b_idx, best_idx] >= MATCH_THRESHOLD and not found_mask[best_idx]:
                found_mask[best_idx] = True
                found[item_ids[best_idx]] = {
                    "asin":             buffer_asins[b_idx],
                    "title":            buffer_titles[b_idx],
                    "description_text": buffer_descs[b_idx],
                }

        buffer_asins.clear()
        buffer_titles.clear()
        buffer_descs.clear()

    processed = 0
    for record in dataset:
        # The description field may be nested or a plain string depending on version
        desc = record.get("description", "") or ""
        if isinstance(desc, list):
            desc = " ".join(desc)
        desc = desc.strip()

        title = record.get("title", "") or ""
        asin  = record.get("parent_asin", record.get("asin", ""))

        if len(desc) < MIN_DESC_LEN:
            continue

        buffer_asins.append(asin)
        buffer_titles.append(title)
        buffer_descs.append(desc)
        processed += 1

        if len(buffer_descs) >= BATCH_SIZE:
            flush_buffer()

        n_found = int(found_mask.sum())
        if processed % 5000 == 0:
            print(f"  Scanned {processed:,} candidates | matched {n_found}/{n_target}")

        if n_found == n_target:
            print("  All items matched — stopping stream early.")
            break

    flush_buffer()   # handle leftover buffer

    n_found = int(found_mask.sum())
    print(f"\nMatching complete: {n_found}/{n_target} items recovered.")

    if n_found < n_target:
        missing = [item_ids[i] for i in range(n_target) if not found_mask[i]]
        print(f"  WARNING: {len(missing)} items not recovered: {missing[:20]} ...")

    # Build output DataFrame
    rows = [
        {"item_id": iid, **meta}
        for iid, meta in found.items()
    ]
    out_df = pd.DataFrame(rows, columns=["item_id", "asin", "title", "description_text"])
    out_df.to_pickle(str(OUT_PKL))
    print(f"Saved item metadata to {OUT_PKL}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
