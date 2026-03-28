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
import json
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATASET_REPO   = "McAuley-Lab/Amazon-Reviews-2023"
METADATA_NAME  = "raw_meta_Movies_and_TV"
SBERT_MODEL    = "all-MiniLM-L6-v2"
MIN_DESC_LEN   = 500          # characters — same filter the authors used
MATCH_THRESHOLD = 0.85        # cosine similarity threshold (lowered to recover more items)
BATCH_SIZE     = 256          # SBERT encoding batch size
DEBUG_TOP_K    = 5            # show top K matches for diagnostics

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

    print(f"Loading Movies & TV metadata ...")
    
    # Check for local file first
    local_meta_file = Path(__file__).resolve().parents[1] / "data" / "meta_Movies_and_TV.jsonl.gz"
    
    if local_meta_file.exists():
        print(f"  Found local metadata file: {local_meta_file}")
        print(f"  Loading JSON lines from local file...")
        records = []
        with gzip.open(local_meta_file, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        print(f"  Loaded {len(records)} records from local file.")
    else:
        print(f"  Local file not found, attempting download from Hugging Face...")
        
        # Direct download URL for the parquet file
        hf_url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw_meta_Movies_and_TV/default/0000.parquet"
        
        try:
            import pyarrow.parquet as pq
            print(f"  Downloading parquet file from HF...")
            resp = requests.get(hf_url, stream=True, timeout=60)
            resp.raise_for_status()
            
            # Buffer to temporary file then read
            temp_file = "/tmp/amazon_meta.parquet"
            with open(temp_file, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            
            table = pq.read_table(temp_file)
            records = table.to_pylist()
            print(f"  Loaded {len(records)} records from parquet.")
            
        except Exception as e:
            print(f"  Parquet failed ({e}), trying JSON fallback...")
            # Fallback: try JSON lines endpoint
            hf_url_json = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/raw/main/raw_meta_Movies_and_TV.jsonl.gz"
            print(f"  Downloading JSON lines from HF...")
            resp = requests.get(hf_url_json, stream=True, timeout=60)
            resp.raise_for_status()
            
            records = []
            with gzip.open(resp.raw, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            print(f"  Loaded {len(records)} records from JSON lines.")
    
    # Iterate through records
    dataset = iter(records)

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
            best_sim = float(sims[b_idx, best_idx])
            
            # Debug: show top matches on first batch
            if processed < BATCH_SIZE and b_idx < 2:
                top_k_sims = np.argsort(sims[b_idx])[-DEBUG_TOP_K:][::-1]
                top_sims = [float(sims[b_idx, idx]) for idx in top_k_sims]
                print(f"    DEBUG batch[{b_idx}]: top sims = {top_sims}")
            
            if best_sim >= MATCH_THRESHOLD and not found_mask[best_idx]:
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
        # Handle nested or flat description fields
        desc = record.get("description")
        if isinstance(desc, list):
            desc = " ".join(str(d) for d in desc if d)
        else:
            desc = str(desc) if desc else ""
        desc = desc.strip()

        title = str(record.get("title") or "").strip()
        # Try different ASIN field names
        asin = record.get("parent_asin") or record.get("asin")
        if not asin:
            continue

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
            print("  All items matched — stopping early.")
            break

    flush_buffer()

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
