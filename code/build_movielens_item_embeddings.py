import argparse
import json
import pickle
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


SCRIPT_NAME = "build_movielens_item_embeddings.py"


class MeanPoolingTextEncoder:
    def __init__(self, model_name: str, device: torch.device):
        resolved_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
        self.model_name = resolved_name
        print(f"Loading text encoder: {resolved_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        self.model = AutoModel.from_pretrained(resolved_name).to(device)
        self.model.eval()
        self.device = device

    def encode(
        self,
        texts: List[str],
        batch_size: int = 256,
        max_length: int = 256,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not texts:
            hidden = getattr(self.model.config, "hidden_size", 384)
            return np.zeros((0, hidden), dtype=np.float32)

        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                model_out = self.model(**enc)
            token_emb = model_out.last_hidden_state
            attention_mask = enc["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
            pooled = (token_emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            if normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
            if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(texts):
                print(f"  Encoded {min(start + batch_size, len(texts))}/{len(texts)} item texts")
        return np.concatenate(outputs, axis=0)


def clean_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    text = re.sub(r"\s+", " ", text)
    return text


def join_nonempty(parts: List[Optional[str]], sep: str = " | ") -> str:
    return sep.join([p for p in parts if p])


def build_item_text(row: pd.Series) -> str:
    title = clean_text(row.get("title")) or f"Movie {int(row['movieId'])}"
    genres_movies = clean_text(row.get("genres_movies"))
    genres_meta = clean_text(row.get("genres_meta"))
    genres = genres_meta or genres_movies
    director = clean_text(row.get("director"))
    cast = clean_text(row.get("cast"))
    writer = clean_text(row.get("writer"))

    parts = [f'Title: {title}']
    if genres:
        parts.append(f"Genres: {genres.replace('|', ', ')}")
    if director:
        parts.append(f"Director: {director}")
    if cast:
        parts.append(f"Cast: {cast}")
    if writer:
        parts.append(f"Writer: {writer}")
    return ". ".join(parts) + "."


def build_history_brief(row: pd.Series) -> str:
    title = clean_text(row.get("title")) or f"Movie {int(row['movieId'])}"
    genres_movies = clean_text(row.get("genres_movies"))
    genres_meta = clean_text(row.get("genres_meta"))
    genres = genres_meta or genres_movies
    director = clean_text(row.get("director"))
    return join_nonempty([
        f'"{title}"',
        f"[{genres.replace('|', ', ')}]" if genres else None,
        f"Dir: {director}" if director else None,
    ])


def load_metadata(id_mapping_path: str, movies_csv_path: str, meta_info_path: str) -> pd.DataFrame:
    id_map = pd.read_csv(id_mapping_path)
    movies = pd.read_csv(movies_csv_path)
    meta = pd.read_csv(meta_info_path, sep=";")

    movies = movies.rename(columns={"genres": "genres_movies"})
    meta = meta.rename(columns={"genres": "genres_meta"})

    merged = id_map.merge(movies, on="movieId", how="left")
    merged = merged.merge(meta, on="movieId", how="left")
    merged["title"] = merged["title"].fillna(merged["movieId"].map(lambda x: f"Movie {x}"))
    merged["item_text"] = merged.apply(build_item_text, axis=1)
    merged["history_brief"] = merged.apply(build_history_brief, axis=1)
    return merged


def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Build MovieLens item embeddings from metadata CSVs.")
    parser.add_argument("--id_mapping_path", type=str, required=True)
    parser.add_argument("--movies_csv_path", type=str, required=True)
    parser.add_argument("--meta_info_path", type=str, required=True)
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = Path(args.output_path) if args.output_path else Path(args.id_mapping_path).resolve().parent / f"movielens_item_embeddings_{args.sbert_model.split('/')[-1]}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.id_mapping_path, args.movies_csv_path, args.meta_info_path)
    print(f"Loaded metadata for {len(metadata)} encoded items")

    encoder = MeanPoolingTextEncoder(args.sbert_model, device=device)
    embeddings = encoder.encode(metadata["item_text"].tolist(), batch_size=args.batch_size, max_length=args.max_length)

    out_df = metadata[["encoded_item_id", "movieId", "title", "genres_movies", "genres_meta", "director", "cast", "writer", "item_text", "history_brief"]].copy()
    out_df = out_df.rename(columns={"encoded_item_id": "item_id"})
    out_df["description"] = [emb.astype(np.float32) for emb in embeddings]

    save_pickle(out_df, output_path)
    print(f"Saved item embeddings to {output_path}")

    summary = {
        "script_name": SCRIPT_NAME,
        "output_path": str(output_path),
        "num_items": int(len(out_df)),
        "sbert_model": args.sbert_model,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "execution_time_sec": time.time() - t0,
    }
    summary_path = output_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
