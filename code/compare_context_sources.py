import argparse
import json
import pickle
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from context_rec import (
    DATASET as DEFAULT_DATASET,
    RecSysModel,
    evaluate_ranking,
    evaluate_ranking_with_context,
    extract_profile_tag,
    load_interactions_csv,
    load_pickle,
    resolve_requested_or_latest_user_profile_embedding_file,
)


PairKey = Tuple[int, int]


@dataclass
class LoadedContextSource:
    label: str
    path: str
    context_dict: Dict[PairKey, List[np.ndarray]]
    pair_count: int
    total_contexts: int
    user_col_used: Optional[str] = None
    item_col_used: Optional[str] = None
    text_mode_used: Optional[str] = None


class MeanPoolingTextEncoder:
    """
    Sentence embedding fallback that stays in the same model family as the repo's
    `all-MiniLM-L6-v2` setup, but avoids depending on `sentence_transformers`.
    """

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
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

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

        return np.concatenate(outputs, axis=0)


def normalize_embedding_list(value) -> List[np.ndarray]:
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [value.astype(np.float32)]
        if value.ndim == 2:
            return [row.astype(np.float32) for row in value]

    if isinstance(value, (list, tuple)):
        if not value:
            return []
        first = value[0]
        if isinstance(first, np.ndarray):
            return [np.asarray(v, dtype=np.float32) for v in value]
        if isinstance(first, (list, tuple)) and not isinstance(first, str):
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 2:
                return [row.astype(np.float32) for row in arr]
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 1:
            return [arr.astype(np.float32)]
        if arr.ndim == 2:
            return [row.astype(np.float32) for row in arr]

    raise TypeError(f"Unsupported embedding container type: {type(value)}")


SITUATION_RE = re.compile(r"SITUATION:\s*(.*?)(?:\n+\s*HISTORY_LINK:|\Z)", re.IGNORECASE | re.DOTALL)


def clean_context_text(text: str, mode: str = "auto") -> Optional[str]:
    if text is None:
        return None

    cleaned = str(text).strip()
    if not cleaned:
        return None
    if cleaned.upper() == "NO_CONTEXT":
        return None

    if mode in {"auto", "situation"}:
        match = SITUATION_RE.search(cleaned)
        if match:
            cleaned = match.group(1).strip()
        elif mode == "situation":
            return None

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def encode_text_context_dict(
    text_context_dict: Dict[PairKey, List[str]],
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
) -> Dict[PairKey, List[np.ndarray]]:
    flat_texts: List[str] = []
    flat_meta: List[Tuple[int, int]] = []
    for pair, texts in text_context_dict.items():
        for text in texts:
            flat_texts.append(text)
            flat_meta.append(pair)

    all_embs = encoder.encode(flat_texts, batch_size=encode_batch_size)
    emb_dict: Dict[PairKey, List[np.ndarray]] = {}
    for pair, emb in zip(flat_meta, all_embs):
        emb_dict.setdefault(pair, []).append(emb.astype(np.float32))
    return emb_dict


def score_column_choice(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    filtered_test_pairs: set,
    available_users: set,
    available_items: set,
) -> Tuple[int, int, int]:
    pairs = set(
        zip(
            df[user_col].astype(int).tolist(),
            df[item_col].astype(int).tolist(),
        )
    )
    pair_coverage = len(pairs & filtered_test_pairs)
    user_coverage = len(set(df[user_col].astype(int).tolist()) & available_users)
    item_coverage = len(set(df[item_col].astype(int).tolist()) & available_items)
    return pair_coverage, user_coverage, item_coverage


def choose_parquet_columns(
    df: pd.DataFrame,
    filtered_test_pairs: set,
    available_users: set,
    available_items: set,
    requested_user_col: str = "auto",
    requested_item_col: str = "auto",
) -> Tuple[str, str]:
    user_candidates = []
    item_candidates = []

    for col in ["user_id", "original_user_id"]:
        if col in df.columns:
            user_candidates.append(col)
    for col in ["target_item", "original_gt_item", "item_id"]:
        if col in df.columns:
            item_candidates.append(col)

    if requested_user_col != "auto":
        if requested_user_col not in df.columns:
            raise KeyError(f"Requested parquet user column '{requested_user_col}' not found")
        user_candidates = [requested_user_col]

    if requested_item_col != "auto":
        if requested_item_col not in df.columns:
            raise KeyError(f"Requested parquet item column '{requested_item_col}' not found")
        item_candidates = [requested_item_col]

    if not user_candidates:
        raise KeyError(
            f"Could not find a usable user column. Available columns: {df.columns.tolist()}"
        )
    if not item_candidates:
        raise KeyError(
            f"Could not find a usable item column. Available columns: {df.columns.tolist()}"
        )

    best_choice = None
    best_score = None
    for user_col in user_candidates:
        for item_col in item_candidates:
            score = score_column_choice(
                df=df,
                user_col=user_col,
                item_col=item_col,
                filtered_test_pairs=filtered_test_pairs,
                available_users=available_users,
                available_items=available_items,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_choice = (user_col, item_col)

    assert best_choice is not None
    print(
        f"Selected parquet columns user='{best_choice[0]}' item='{best_choice[1]}' "
        f"with pair/user/item coverage={best_score}"
    )
    return best_choice


def load_standard_or_text_pickle_contexts(
    obj,
    label: str,
    path: str,
    encoder: Optional[MeanPoolingTextEncoder],
    encode_batch_size: int,
    text_mode: str,
) -> LoadedContextSource:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict pickle for {path}, got {type(obj)}")

    sample_key = next(iter(obj), None)
    if sample_key is None:
        return LoadedContextSource(label=label, path=path, context_dict={}, pair_count=0, total_contexts=0)
    if not (isinstance(sample_key, tuple) and len(sample_key) == 2):
        raise TypeError(
            f"Expected keys like (user_id, item_id) in {path}; got sample key {sample_key!r}"
        )

    sample_value = obj[sample_key]

    is_embedding_like = False
    if isinstance(sample_value, np.ndarray):
        is_embedding_like = True
    elif isinstance(sample_value, (list, tuple)) and sample_value:
        first = sample_value[0]
        is_embedding_like = isinstance(first, np.ndarray) or (
            isinstance(first, (list, tuple)) and not isinstance(first, str)
        )

    if is_embedding_like:
        context_dict = {
            (int(k[0]), int(k[1])): normalize_embedding_list(v)
            for k, v in obj.items()
        }
        total_contexts = sum(len(v) for v in context_dict.values())
        return LoadedContextSource(
            label=label,
            path=path,
            context_dict=context_dict,
            pair_count=len(context_dict),
            total_contexts=total_contexts,
        )

    if encoder is None:
        raise RuntimeError(
            f"{path} appears to contain text contexts, but no text encoder was initialized."
        )

    text_context_dict: Dict[PairKey, List[str]] = {}
    for key, value in obj.items():
        pair = (int(key[0]), int(key[1]))
        texts = value if isinstance(value, (list, tuple)) else [value]
        cleaned_texts = [clean_context_text(t, mode=text_mode) for t in texts]
        cleaned_texts = [t for t in cleaned_texts if t]
        if cleaned_texts:
            text_context_dict[pair] = dedupe_preserve_order(cleaned_texts)

    context_dict = encode_text_context_dict(text_context_dict, encoder, encode_batch_size)
    total_contexts = sum(len(v) for v in context_dict.values())
    return LoadedContextSource(
        label=label,
        path=path,
        context_dict=context_dict,
        pair_count=len(context_dict),
        total_contexts=total_contexts,
        text_mode_used=text_mode,
    )


def load_parquet_contexts(
    path: str,
    label: str,
    filtered_test_pairs: set,
    available_users: set,
    available_items: set,
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
    parquet_user_id_col: str,
    parquet_item_id_col: str,
    text_mode: str,
) -> LoadedContextSource:
    df = pd.read_parquet(path)
    if "generated_context" not in df.columns:
        raise KeyError(
            f"Expected a 'generated_context' column in {path}. Available columns: {df.columns.tolist()}"
        )

    user_col, item_col = choose_parquet_columns(
        df=df,
        filtered_test_pairs=filtered_test_pairs,
        available_users=available_users,
        available_items=available_items,
        requested_user_col=parquet_user_id_col,
        requested_item_col=parquet_item_id_col,
    )

    text_context_dict: Dict[PairKey, List[str]] = OrderedDict()
    for row in df.itertuples(index=False):
        user_id = int(getattr(row, user_col))
        item_id = int(getattr(row, item_col))
        text = clean_context_text(getattr(row, "generated_context"), mode=text_mode)
        if text is None:
            continue
        text_context_dict.setdefault((user_id, item_id), []).append(text)

    text_context_dict = {
        pair: dedupe_preserve_order(texts)
        for pair, texts in text_context_dict.items()
        if texts
    }

    context_dict = encode_text_context_dict(text_context_dict, encoder, encode_batch_size)
    total_contexts = sum(len(v) for v in context_dict.values())
    return LoadedContextSource(
        label=label,
        path=path,
        context_dict=context_dict,
        pair_count=len(context_dict),
        total_contexts=total_contexts,
        user_col_used=user_col,
        item_col_used=item_col,
        text_mode_used=text_mode,
    )


def auto_label_for_path(path: str) -> str:
    return Path(path).stem


def load_context_source(
    path: str,
    label: str,
    filtered_test_pairs: set,
    available_users: set,
    available_items: set,
    encoder: Optional[MeanPoolingTextEncoder],
    encode_batch_size: int,
    parquet_user_id_col: str,
    parquet_item_id_col: str,
    text_mode: str,
) -> LoadedContextSource:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        if encoder is None:
            raise RuntimeError(f"A text encoder is required to load parquet contexts from {path}")
        return load_parquet_contexts(
            path=path,
            label=label,
            filtered_test_pairs=filtered_test_pairs,
            available_users=available_users,
            available_items=available_items,
            encoder=encoder,
            encode_batch_size=encode_batch_size,
            parquet_user_id_col=parquet_user_id_col,
            parquet_item_id_col=parquet_item_id_col,
            text_mode=text_mode,
        )

    if suffix in {".pkl", ".pickle"}:
        obj = load_pickle(path)
        return load_standard_or_text_pickle_contexts(
            obj=obj,
            label=label,
            path=path,
            encoder=encoder,
            encode_batch_size=encode_batch_size,
            text_mode=text_mode,
        )

    raise ValueError(f"Unsupported context source format for {path}")


def compute_context_coverage(filtered_test_pairs: set, context_dict: Dict[PairKey, List[np.ndarray]]) -> Tuple[int, int]:
    covered_pairs = sum(1 for pair in filtered_test_pairs if pair in context_dict)
    return covered_pairs, len(filtered_test_pairs)




def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
        return model
    except RuntimeError:
        pass

    if any(str(k).startswith("module.") for k in state_dict.keys()):
        stripped = {str(k)[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(stripped)
        return model

    prefixed = {f"module.{k}": v for k, v in state_dict.items()}
    wrapped_model = torch.nn.DataParallel(model).to(device)
    wrapped_model.load_state_dict(prefixed)
    return wrapped_model


def resolve_checkpoint_path(script_dir: Path, dataset: str, seed: Optional[int], explicit_checkpoint: Optional[str]) -> Path:
    search_roots = [script_dir, script_dir.parent, Path.cwd()]

    if explicit_checkpoint:
        cp = Path(explicit_checkpoint)
        if cp.is_absolute() and cp.exists():
            return cp
        for root in search_roots:
            candidate = (root / explicit_checkpoint).resolve()
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Checkpoint not found: {explicit_checkpoint}")

    if seed is not None:
        for root in search_roots:
            candidate = root / f"best_context_rec_{dataset}_seed_{seed}.pt"
            if candidate.exists():
                return candidate.resolve()

    matches = []
    for root in search_roots:
        matches.extend(root.glob(f"best_context_rec_{dataset}_seed_*.pt"))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime).resolve()

    raise FileNotFoundError(
        "Could not find a ContextRec checkpoint automatically. "
        "Pass --checkpoint_path or run context_rec.py once to create one."
    )


def build_rank_report(
    model,
    user_representations: Dict[int, np.ndarray],
    item_dict: Dict[int, np.ndarray],
    test_df: pd.DataFrame,
    device: torch.device,
    source_label: str,
) -> pd.DataFrame:
    model.eval()

    user_to_test_items: Dict[int, set] = {}
    for row in test_df.itertuples(index=False):
        user_to_test_items.setdefault(int(row.user_id), set()).add(int(row.item_id))

    all_item_ids = list(item_dict.keys())
    item_embs_torch = torch.tensor(
        np.stack([item_dict[i].astype(np.float32) for i in all_item_ids], axis=0)
    ).to(device)

    rows = []
    for user_id, test_items in user_to_test_items.items():
        if user_id not in user_representations:
            continue

        user_vec = torch.tensor(
            np.asarray(user_representations[user_id], dtype=np.float32)
        ).unsqueeze(0).to(device)

        user_rep_expanded = user_vec.repeat(item_embs_torch.size(0), 1)
        concat_vec = torch.cat([user_rep_expanded, item_embs_torch], dim=1)
        with torch.no_grad():
            fc = model.module.fc if isinstance(model, torch.nn.DataParallel) else model.fc
            logits = fc(concat_vec)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        ranked_idx = np.argsort(-probs)
        item_to_rank = {all_item_ids[idx]: rank + 1 for rank, idx in enumerate(ranked_idx.tolist())}
        item_to_score = {all_item_ids[idx]: float(probs[idx]) for idx in range(len(all_item_ids))}

        for item_id in test_items:
            rows.append(
                {
                    "source": source_label,
                    "user_id": user_id,
                    "item_id": item_id,
                    "rank": item_to_rank[item_id],
                    "score": item_to_score[item_id],
                    "hit_at_10": int(item_to_rank[item_id] <= 10),
                    "hit_at_20": int(item_to_rank[item_id] <= 20),
                }
            )

    return pd.DataFrame(rows)


def build_avg_context_user_representations(
    user_dict: Dict[int, np.ndarray],
    test_df: pd.DataFrame,
    context_dict: Dict[PairKey, List[np.ndarray]],
    alpha: float,
) -> Dict[int, np.ndarray]:
    user_to_test_items: Dict[int, set] = {}
    for row in test_df.itertuples(index=False):
        user_to_test_items.setdefault(int(row.user_id), set()).add(int(row.item_id))

    fused_user_dict = {}
    for user_id, test_items in user_to_test_items.items():
        if user_id not in user_dict:
            continue
        user_long_np = np.asarray(user_dict[user_id], dtype=np.float32)
        all_ctx_embs = [emb for item_id in test_items for emb in context_dict.get((user_id, item_id), [])]
        if not all_ctx_embs:
            continue
        mean_ctx = np.mean(np.stack(all_ctx_embs, axis=0), axis=0).astype(np.float32)
        fused_user_dict[user_id] = alpha * user_long_np + (1 - alpha) * mean_ctx
    return fused_user_dict


def save_summary_json(output_path: Path, payload: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved comparison summary to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ranking results across multiple context sources at inference time."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--llm_model", type=str, default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--user_profile_file",
        type=str,
        default=None,
        help=(
            "Embedding pickle to use for long-term user descriptions, either absolute or relative to data/<dataset>. "
            "If omitted, the newest matching embedding pickle is used."
        ),
    )
    parser.add_argument("--context_path", action="append", default=[], help="Can be passed multiple times.")
    parser.add_argument("--context_label", action="append", default=[], help="Optional labels aligned with --context_path.")
    parser.add_argument("--include_default_context", action="store_true", help="Also compare against data/<dataset>/bert_context_profiles_test.pkl if present.")
    parser.add_argument("--context_alpha", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--parquet_user_id_col", type=str, default="auto", choices=["auto", "user_id", "original_user_id"])
    parser.add_argument("--parquet_item_id_col", type=str, default="auto", choices=["auto", "target_item", "original_gt_item", "item_id"])
    parser.add_argument("--text_mode", type=str, default="auto", choices=["auto", "full", "situation"])
    parser.add_argument("--save_rank_reports", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    script_dir = Path(__file__).resolve().parent
    root = script_dir.parents[0]
    data_dir = root / "data" / args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = resolve_checkpoint_path(script_dir, args.dataset, args.seed, args.checkpoint_path)
    print(f"Using checkpoint: {checkpoint_path}")

    item_embeddings_path = data_dir / "bert_item_features.pkl"
    user_embeddings_path = Path(
        resolve_requested_or_latest_user_profile_embedding_file(
            data_dir=data_dir,
            model_name_short=args.llm_model.split("/")[-1],
            explicit_path=args.user_profile_file,
        )
    )
    profile_tag = extract_profile_tag(str(user_embeddings_path))
    test_path = data_dir / "test.csv"

    print(f"Loading item embeddings from: {item_embeddings_path}")
    print(f"Loading user embeddings from: {user_embeddings_path}")
    print(f"User profile tag: {profile_tag}")
    print(f"Loading test split from: {test_path}")

    item_df = load_pickle(str(item_embeddings_path))
    user_df = load_pickle(str(user_embeddings_path))
    test_df = load_interactions_csv(str(test_path))

    item_dict = {int(row["item_id"]): np.asarray(row["description"], dtype=np.float32) for _, row in item_df.iterrows()}
    user_dict = {int(row["user_id"]): np.asarray(row["profile"], dtype=np.float32) for _, row in user_df.iterrows()}

    available_users = set(user_dict.keys())
    available_items = set(item_dict.keys())
    test_df = test_df[
        test_df["user_id"].isin(available_users) & test_df["item_id"].isin(available_items)
    ].copy()
    filtered_test_pairs = {(int(r.user_id), int(r.item_id)) for r in test_df.itertuples(index=False)}

    print(
        f"Filtered test size: {len(test_df)} rows across "
        f"{test_df['user_id'].nunique()} users and {test_df['item_id'].nunique()} items"
    )

    model = RecSysModel(embed_dim=384, hidden_dim=128, dropout=0.2).to(device)
    model = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    top_k_list = sorted(set(args.top_k))
    baseline_metrics = evaluate_ranking(
        model=model,
        user_dict=user_dict,
        item_dict=item_dict,
        test_df=test_df,
        device=device,
        top_k_list=top_k_list,
    )
    print("\nBaseline ranking metrics")
    for k in top_k_list:
        print(
            f"  Top-{k}: precision={baseline_metrics[k]['precision']:.4f} "
            f"recall={baseline_metrics[k]['recall']:.4f} ndcg={baseline_metrics[k]['ndcg']:.4f}"
        )

    rank_reports = {}
    if args.save_rank_reports:
        rank_reports["baseline"] = build_rank_report(
            model=model,
            user_representations=user_dict,
            item_dict=item_dict,
            test_df=test_df,
            device=device,
            source_label="baseline",
        )

    context_inputs: List[Tuple[str, str]] = []
    if args.include_default_context:
        default_context_path = data_dir / "bert_context_profiles_test.pkl"
        if default_context_path.exists():
            context_inputs.append((str(default_context_path), "my_generated_contexts"))
        else:
            print(f"Default context pickle not found at {default_context_path}; skipping it.")

    for idx, path in enumerate(args.context_path):
        label = args.context_label[idx] if idx < len(args.context_label) else auto_label_for_path(path)
        context_inputs.append((path, label))

    if not context_inputs:
        default_context_path = data_dir / "bert_context_profiles_test.pkl"
        if default_context_path.exists():
            context_inputs.append((str(default_context_path), "my_generated_contexts"))
        else:
            print("No context sources were provided, and the default context pickle was not found.")

    encoder = None
    if any(Path(path).suffix.lower() in {".parquet"} for path, _ in context_inputs):
        encoder = MeanPoolingTextEncoder(args.sbert_model, device=device)

    comparison_results = OrderedDict()
    results_dir = script_dir / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for path, label in context_inputs:
        print(f"\nLoading context source: {label} ({path})")
        source = load_context_source(
            path=path,
            label=label,
            filtered_test_pairs=filtered_test_pairs,
            available_users=available_users,
            available_items=available_items,
            encoder=encoder,
            encode_batch_size=args.encode_batch_size,
            parquet_user_id_col=args.parquet_user_id_col,
            parquet_item_id_col=args.parquet_item_id_col,
            text_mode=args.text_mode,
        )

        covered_pairs, total_pairs = compute_context_coverage(filtered_test_pairs, source.context_dict)
        print(
            f"  Pairs loaded: {source.pair_count} | contexts loaded: {source.total_contexts} | "
            f"coverage on filtered test pairs: {covered_pairs}/{total_pairs}"
        )

        metrics = evaluate_ranking_with_context(
            model=model,
            user_dict=user_dict,
            item_dict=item_dict,
            test_df=test_df,
            context_dict=source.context_dict,
            alpha=args.context_alpha,
            device=device,
            top_k_list=top_k_list,
        )

        print(f"  {label} avg-fusion metrics")
        for k in top_k_list:
            print(
                f"    Top-{k}: precision={metrics['avg'][k]['precision']:.4f} "
                f"recall={metrics['avg'][k]['recall']:.4f} ndcg={metrics['avg'][k]['ndcg']:.4f}"
            )
        print(f"  {label} oracle-best metrics")
        for k in top_k_list:
            print(
                f"    Top-{k}: precision={metrics['best'][k]['precision']:.4f} "
                f"recall={metrics['best'][k]['recall']:.4f} ndcg={metrics['best'][k]['ndcg']:.4f}"
            )

        delta_vs_baseline = {
            str(k): {
                metric: metrics["avg"][k][metric] - baseline_metrics[k][metric]
                for metric in ["precision", "recall", "ndcg"]
            }
            for k in top_k_list
        }

        result_entry = {
            "path": path,
            "pair_count": source.pair_count,
            "total_contexts": source.total_contexts,
            "covered_test_pairs": covered_pairs,
            "total_test_pairs": total_pairs,
            "avg": metrics["avg"],
            "best": metrics["best"],
            "delta_vs_baseline_avg": delta_vs_baseline,
            "user_col_used": source.user_col_used,
            "item_col_used": source.item_col_used,
            "text_mode_used": source.text_mode_used,
        }

        if args.save_rank_reports:
            avg_user_reps = build_avg_context_user_representations(
                user_dict=user_dict,
                test_df=test_df,
                context_dict=source.context_dict,
                alpha=args.context_alpha,
            )
            avg_rank_report = build_rank_report(
                model=model,
                user_representations=avg_user_reps,
                item_dict=item_dict,
                test_df=test_df,
                device=device,
                source_label=label,
            )
            if not avg_rank_report.empty and "baseline" in rank_reports:
                merged = rank_reports["baseline"][["user_id", "item_id", "rank"]].rename(
                    columns={"rank": "baseline_rank"}
                ).merge(
                    avg_rank_report[["user_id", "item_id", "rank", "score", "hit_at_10", "hit_at_20"]].rename(
                        columns={
                            "rank": "context_avg_rank",
                            "score": "context_avg_score",
                            "hit_at_10": "context_avg_hit_at_10",
                            "hit_at_20": "context_avg_hit_at_20",
                        }
                    ),
                    on=["user_id", "item_id"],
                    how="inner",
                )
                merged["rank_delta_improved_if_positive"] = merged["baseline_rank"] - merged["context_avg_rank"]
                report_path = results_dir / f"compare_contexts_{args.dataset}_{profile_tag}_{label}_{timestamp}_rank_report.csv"
                merged.to_csv(report_path, index=False)
                result_entry["avg_rank_report_csv"] = str(report_path)
                print(f"  Saved rank report to {report_path}")

        comparison_results[label] = result_entry

    summary = {
        "dataset": args.dataset,
        "llm_model": args.llm_model,
        "checkpoint_path": str(checkpoint_path),
        "user_profile_file": str(user_embeddings_path),
        "profile_tag": profile_tag,
        "context_alpha": args.context_alpha,
        "top_k": top_k_list,
        "baseline": baseline_metrics,
        "context_sources": comparison_results,
        "execution_time_sec": time.time() - t0,
    }

    out_json = results_dir / f"compare_contexts_{args.dataset}_{profile_tag}_{timestamp}.json"
    save_summary_json(out_json, summary)
    print("Done.")


if __name__ == "__main__":
    main()
