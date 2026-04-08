import argparse
import json
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
    load_pickle,
    resolve_requested_or_latest_user_profile_embedding_file,
)


PairKey = Tuple[int, int]
CaseKey = Tuple[int, int, int]


@dataclass
class LoadedContextSource:
    label: str
    path: str
    pair_contexts: Dict[PairKey, List[np.ndarray]]
    case_contexts: Dict[CaseKey, List[np.ndarray]]
    pair_count: int
    case_count: int
    total_contexts: int
    pair_history_embeddings: Dict[PairKey, np.ndarray]
    case_history_embeddings: Dict[CaseKey, np.ndarray]
    pair_history_texts: Dict[PairKey, str]
    case_history_texts: Dict[CaseKey, str]
    pair_context_texts: Dict[PairKey, List[str]]
    case_context_texts: Dict[CaseKey, List[str]]
    user_col_used: Optional[str] = None
    item_col_used: Optional[str] = None
    step_col_used: Optional[str] = None
    context_text_col_used: Optional[str] = None
    history_text_col_used: Optional[str] = None
    text_mode_used: Optional[str] = None


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


SITUATION_RE = re.compile(
    r"SITUATION:\s*(.*?)(?:\n+\s*(?:TRIGGER|HISTORY_LINK):|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def clean_text(text: str) -> Optional[str]:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def clean_context_text(text: str, mode: str = "full") -> Optional[str]:
    cleaned = clean_text(text)
    if cleaned is None:
        return None
    if cleaned.upper() == "NO_CONTEXT":
        return None

    if mode in {"auto", "situation"}:
        match = SITUATION_RE.search(cleaned)
        if match:
            cleaned = clean_text(match.group(1))
        elif mode == "situation":
            return None

    return cleaned


def dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def encode_grouped_lists(
    grouped_texts: Dict[Tuple[int, ...], List[str]],
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
) -> Dict[Tuple[int, ...], List[np.ndarray]]:
    flat_texts: List[str] = []
    flat_keys: List[Tuple[int, ...]] = []
    for key, texts in grouped_texts.items():
        for text in texts:
            flat_texts.append(text)
            flat_keys.append(key)

    all_embs = encoder.encode(flat_texts, batch_size=encode_batch_size)
    out: Dict[Tuple[int, ...], List[np.ndarray]] = {}
    for key, emb in zip(flat_keys, all_embs):
        out.setdefault(key, []).append(emb.astype(np.float32))
    return out


def encode_grouped_single_texts(
    grouped_texts: Dict[Tuple[int, ...], str],
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
) -> Dict[Tuple[int, ...], np.ndarray]:
    if not grouped_texts:
        return {}
    keys = list(grouped_texts.keys())
    texts = [grouped_texts[k] for k in keys]
    embs = encoder.encode(texts, batch_size=encode_batch_size)
    return {key: emb.astype(np.float32) for key, emb in zip(keys, embs)}


def choose_context_columns(
    df: pd.DataFrame,
    requested_user_col: str,
    requested_item_col: str,
    requested_step_col: str,
    requested_context_text_col: str,
    requested_history_text_col: str,
) -> Tuple[str, str, Optional[str], str, Optional[str]]:
    user_candidates = [c for c in ["original_user_id", "user_id"] if c in df.columns]
    item_candidates = [c for c in ["original_gt_item", "target_item", "item_id", "target_x"] if c in df.columns]
    step_candidates = [c for c in ["step"] if c in df.columns]
    context_text_candidates = [c for c in ["generated_context"] if c in df.columns]
    history_text_candidates = [c for c in ["user_prompt"] if c in df.columns]

    if requested_user_col != "auto":
        if requested_user_col not in df.columns:
            raise KeyError(f"Requested user column '{requested_user_col}' not found")
        user_candidates = [requested_user_col]
    if requested_item_col != "auto":
        if requested_item_col not in df.columns:
            raise KeyError(f"Requested item column '{requested_item_col}' not found")
        item_candidates = [requested_item_col]
    if requested_step_col != "auto":
        if requested_step_col == "none":
            step_candidates = []
        elif requested_step_col not in df.columns:
            raise KeyError(f"Requested step column '{requested_step_col}' not found")
        else:
            step_candidates = [requested_step_col]
    if requested_context_text_col != "auto":
        if requested_context_text_col not in df.columns:
            raise KeyError(f"Requested context text column '{requested_context_text_col}' not found")
        context_text_candidates = [requested_context_text_col]
    if requested_history_text_col != "auto":
        if requested_history_text_col == "none":
            history_text_candidates = []
        elif requested_history_text_col not in df.columns:
            raise KeyError(f"Requested history text column '{requested_history_text_col}' not found")
        else:
            history_text_candidates = [requested_history_text_col]

    if not user_candidates:
        raise KeyError(f"No usable user column in context file. Available: {df.columns.tolist()}")
    if not item_candidates:
        raise KeyError(f"No usable item column in context file. Available: {df.columns.tolist()}")
    if not context_text_candidates:
        raise KeyError(f"No usable context text column in context file. Available: {df.columns.tolist()}")

    user_col = user_candidates[0]
    item_col = item_candidates[0]
    step_col = step_candidates[0] if step_candidates else None
    context_text_col = context_text_candidates[0]
    history_text_col = history_text_candidates[0] if history_text_candidates else None
    print(
        "Selected context columns "
        f"user='{user_col}' item='{item_col}' step='{step_col}' "
        f"context_text='{context_text_col}' history_text='{history_text_col}'"
    )
    return user_col, item_col, step_col, context_text_col, history_text_col


def load_context_parquet(
    path: str,
    label: str,
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
    user_col: str,
    item_col: str,
    step_col: str,
    context_text_col: str,
    history_text_col: str,
    text_mode: str,
) -> LoadedContextSource:
    df = pd.read_parquet(path)

    user_col, item_col, step_col, context_text_col, history_text_col = choose_context_columns(
        df=df,
        requested_user_col=user_col,
        requested_item_col=item_col,
        requested_step_col=step_col,
        requested_context_text_col=context_text_col,
        requested_history_text_col=history_text_col,
    )

    grouped_case_texts: Dict[CaseKey, List[str]] = OrderedDict()
    grouped_pair_texts: Dict[PairKey, List[str]] = OrderedDict()
    grouped_case_context_raw: Dict[CaseKey, List[str]] = OrderedDict()
    grouped_pair_context_raw: Dict[PairKey, List[str]] = OrderedDict()
    grouped_case_history_raw: Dict[CaseKey, str] = OrderedDict()
    grouped_pair_history_raw: Dict[PairKey, str] = OrderedDict()

    for row in df.itertuples(index=False):
        user_id = int(getattr(row, user_col))
        item_id = int(getattr(row, item_col))
        pair_key = (user_id, item_id)
        step = int(getattr(row, step_col)) if step_col is not None else None
        case_key = (user_id, step, item_id) if step_col is not None else None

        context_text = clean_context_text(getattr(row, context_text_col), mode=text_mode)
        history_text = clean_text(getattr(row, history_text_col)) if history_text_col is not None else None

        if history_text:
            grouped_pair_history_raw.setdefault(pair_key, history_text)
            if case_key is not None:
                grouped_case_history_raw.setdefault(case_key, history_text)

        if context_text is None:
            continue

        grouped_pair_texts.setdefault(pair_key, []).append(context_text)
        grouped_pair_context_raw.setdefault(pair_key, []).append(context_text)
        if case_key is not None:
            grouped_case_texts.setdefault(case_key, []).append(context_text)
            grouped_case_context_raw.setdefault(case_key, []).append(context_text)

    grouped_pair_texts = {k: dedupe_preserve_order(v) for k, v in grouped_pair_texts.items() if v}
    grouped_case_texts = {k: dedupe_preserve_order(v) for k, v in grouped_case_texts.items() if v}
    grouped_pair_context_raw = {k: dedupe_preserve_order(v) for k, v in grouped_pair_context_raw.items() if v}
    grouped_case_context_raw = {k: dedupe_preserve_order(v) for k, v in grouped_case_context_raw.items() if v}

    pair_contexts = encode_grouped_lists(grouped_pair_texts, encoder, encode_batch_size)
    case_contexts = encode_grouped_lists(grouped_case_texts, encoder, encode_batch_size) if grouped_case_texts else {}
    pair_history_embeddings = encode_grouped_single_texts(grouped_pair_history_raw, encoder, encode_batch_size)
    case_history_embeddings = encode_grouped_single_texts(grouped_case_history_raw, encoder, encode_batch_size)
    total_contexts = sum(len(v) for v in pair_contexts.values())

    return LoadedContextSource(
        label=label,
        path=path,
        pair_contexts=pair_contexts,
        case_contexts=case_contexts,
        pair_count=len(pair_contexts),
        case_count=len(case_contexts),
        total_contexts=total_contexts,
        pair_history_embeddings=pair_history_embeddings,
        case_history_embeddings=case_history_embeddings,
        pair_history_texts=grouped_pair_history_raw,
        case_history_texts=grouped_case_history_raw,
        pair_context_texts=grouped_pair_context_raw,
        case_context_texts=grouped_case_context_raw,
        user_col_used=user_col,
        item_col_used=item_col,
        step_col_used=step_col,
        context_text_col_used=context_text_col,
        history_text_col_used=history_text_col,
        text_mode_used=text_mode,
    )


def load_context_source(
    path: str,
    label: str,
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
    parquet_user_id_col: str,
    parquet_item_id_col: str,
    parquet_step_col: str,
    parquet_context_text_col: str,
    parquet_history_text_col: str,
    text_mode: str,
) -> LoadedContextSource:
    suffix = Path(path).suffix.lower()
    if suffix != ".parquet":
        raise ValueError(
            f"This reranking script is now parquet-only. Unsupported context source format: {path}"
        )
    return load_context_parquet(
        path=path,
        label=label,
        encoder=encoder,
        encode_batch_size=encode_batch_size,
        user_col=parquet_user_id_col,
        item_col=parquet_item_id_col,
        step_col=parquet_step_col,
        context_text_col=parquet_context_text_col,
        history_text_col=parquet_history_text_col,
        text_mode=text_mode,
    )


def auto_label_for_path(path: str) -> str:
    return Path(path).stem


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


def parse_topk_list(value) -> List[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        return [int(v) for v in parsed]
    raise ValueError(f"Unsupported top-k candidate value: {value!r}")


def score_candidates(
    model,
    user_rep: np.ndarray,
    candidate_item_ids: List[int],
    item_dict: Dict[int, np.ndarray],
    device: torch.device,
) -> Dict[int, float]:
    item_embs = np.stack([np.asarray(item_dict[i], dtype=np.float32) for i in candidate_item_ids], axis=0)
    item_embs_torch = torch.from_numpy(item_embs).to(device)
    user_torch = torch.from_numpy(np.asarray(user_rep, dtype=np.float32)).unsqueeze(0).to(device)
    user_rep_expanded = user_torch.repeat(item_embs_torch.size(0), 1)
    concat_vec = torch.cat([user_rep_expanded, item_embs_torch], dim=1)
    with torch.no_grad():
        fc = model.module.fc if isinstance(model, torch.nn.DataParallel) else model.fc
        logits = fc(concat_vec)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return {item_id: float(score) for item_id, score in zip(candidate_item_ids, probs.tolist())}


def rank_of_item(sorted_item_ids: List[int], target_item: int) -> int:
    return sorted_item_ids.index(target_item) + 1


def build_hit_metrics(df: pd.DataFrame, rank_col: str, ks=(1, 3, 5, 10)) -> Dict[str, float]:
    if df.empty:
        return {f"hit@{k}": 0.0 for k in ks}
    return {f"hit@{k}": float((df[rank_col] <= k).mean()) for k in ks}


def summarize_case_report(df: pd.DataFrame, before_col: str, after_col: str) -> Dict[str, float]:
    if df.empty:
        return {
            "case_count": 0,
            "avg_rank_before": 0.0,
            "avg_rank_after": 0.0,
            "median_rank_before": 0.0,
            "median_rank_after": 0.0,
            "avg_delta": 0.0,
            "improved_count": 0,
            "unchanged_count": 0,
            "worsened_count": 0,
            **build_hit_metrics(df.assign(**{before_col: []}), before_col),
        }

    delta = df[before_col] - df[after_col]
    summary = {
        "case_count": int(len(df)),
        "avg_rank_before": float(df[before_col].mean()),
        "avg_rank_after": float(df[after_col].mean()),
        "median_rank_before": float(df[before_col].median()),
        "median_rank_after": float(df[after_col].median()),
        "avg_delta": float(delta.mean()),
        "improved_count": int((delta > 0).sum()),
        "unchanged_count": int((delta == 0).sum()),
        "worsened_count": int((delta < 0).sum()),
    }
    summary.update({f"before_{k}": v for k, v in build_hit_metrics(df, before_col).items()})
    summary.update({f"after_{k}": v for k, v in build_hit_metrics(df, after_col).items()})
    return summary


def get_context_embeddings_for_case(source: LoadedContextSource, user_id: int, step: int, item_id: int) -> List[np.ndarray]:
    if (user_id, step, item_id) in source.case_contexts:
        return source.case_contexts[(user_id, step, item_id)]
    return source.pair_contexts.get((user_id, item_id), [])


def get_history_embedding_for_case(source: LoadedContextSource, user_id: int, step: int, item_id: int) -> Optional[np.ndarray]:
    if (user_id, step, item_id) in source.case_history_embeddings:
        return source.case_history_embeddings[(user_id, step, item_id)]
    return source.pair_history_embeddings.get((user_id, item_id))


def get_history_text_for_case(source: LoadedContextSource, user_id: int, step: int, item_id: int) -> Optional[str]:
    if (user_id, step, item_id) in source.case_history_texts:
        return source.case_history_texts[(user_id, step, item_id)]
    return source.pair_history_texts.get((user_id, item_id))


def get_context_texts_for_case(source: LoadedContextSource, user_id: int, step: int, item_id: int) -> List[str]:
    if (user_id, step, item_id) in source.case_context_texts:
        return source.case_context_texts[(user_id, step, item_id)]
    return source.pair_context_texts.get((user_id, item_id), [])


def build_user_representation(
    args,
    source: LoadedContextSource,
    user_dict: Dict[int, np.ndarray],
    user_id: int,
    step: int,
    item_id: int,
) -> Optional[np.ndarray]:
    if args.user_rep_source == "parquet_user_prompt":
        return get_history_embedding_for_case(source, user_id, step, item_id)
    return np.asarray(user_dict[user_id], dtype=np.float32) if user_id in user_dict else None


def rerank_source_against_sasrec(
    model,
    user_dict: Dict[int, np.ndarray],
    item_dict: Dict[int, np.ndarray],
    sasrec_df: pd.DataFrame,
    source: LoadedContextSource,
    args,
    device: torch.device,
    max_cases: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    total = len(sasrec_df) if max_cases is None else min(len(sasrec_df), max_cases)

    for idx, row in enumerate(sasrec_df.itertuples(index=False), start=1):
        if max_cases is not None and len(rows) >= max_cases:
            break
        if idx % 5000 == 0:
            print(f"  Processed {idx}/{total} SASRec rows for {source.label}")

        user_id = int(row.original_user_id)
        step = int(row.step)
        target_item = int(row.target_x)

        if target_item not in item_dict:
            continue

        base_user = build_user_representation(args, source, user_dict, user_id, step, target_item)
        if base_user is None:
            continue

        contexts = get_context_embeddings_for_case(source, user_id, step, target_item)
        if not contexts:
            continue

        topk_items = [int(i) for i in parse_topk_list(row.topk_items)]
        filtered_topk = [i for i in topk_items if i in item_dict]
        candidates = list(filtered_topk)
        if target_item not in candidates:
            candidates.append(target_item)
        if target_item not in candidates:
            continue

        baseline_rank = candidates.index(target_item) + 1
        full_sasrec_rank = None if pd.isna(row.target_rank) else int(row.target_rank)
        target_probability = None if pd.isna(row.target_probability) else float(row.target_probability)
        gt_in_topk = int(target_item in filtered_topk)

        long_only_scores = score_candidates(model, base_user, candidates, item_dict, device)
        long_only_order = sorted(candidates, key=lambda item_id: long_only_scores[item_id], reverse=True)
        long_only_rank = rank_of_item(long_only_order, target_item)

        mean_ctx = np.mean(np.stack(contexts, axis=0), axis=0).astype(np.float32)
        fused_avg = args.context_alpha * base_user + (1 - args.context_alpha) * mean_ctx
        avg_scores = score_candidates(model, fused_avg, candidates, item_dict, device)
        avg_order = sorted(candidates, key=lambda item_id: avg_scores[item_id], reverse=True)
        avg_rank = rank_of_item(avg_order, target_item)

        best_rank = None
        best_score = None
        best_context_text = None
        raw_context_texts = get_context_texts_for_case(source, user_id, step, target_item)
        for ctx_idx, ctx_emb in enumerate(contexts):
            fused = args.context_alpha * base_user + (1 - args.context_alpha) * np.asarray(ctx_emb, dtype=np.float32)
            ctx_scores = score_candidates(model, fused, candidates, item_dict, device)
            ctx_order = sorted(candidates, key=lambda item_id: ctx_scores[item_id], reverse=True)
            ctx_rank = rank_of_item(ctx_order, target_item)
            target_score = ctx_scores[target_item]
            if best_rank is None or ctx_rank < best_rank or (ctx_rank == best_rank and target_score > (best_score or -np.inf)):
                best_rank = ctx_rank
                best_score = target_score
                if ctx_idx < len(raw_context_texts):
                    best_context_text = raw_context_texts[ctx_idx]

        history_text = get_history_text_for_case(source, user_id, step, target_item)
        rows.append(
            {
                "source": source.label,
                "original_user_id": user_id,
                "step": step,
                "target_item": target_item,
                "num_candidates": len(candidates),
                "num_contexts": len(contexts),
                "gt_in_topk": gt_in_topk,
                "sasrec_candidate_rank": baseline_rank,
                "sasrec_full_rank": full_sasrec_rank,
                "sasrec_target_probability": target_probability,
                "contextrec_long_only_rank": long_only_rank,
                "contextrec_avg_rank": avg_rank,
                "contextrec_best_rank": best_rank,
                "delta_long_only_vs_sasrec": baseline_rank - long_only_rank,
                "delta_avg_vs_sasrec": baseline_rank - avg_rank,
                "delta_best_vs_sasrec": baseline_rank - best_rank,
                "history_text": history_text,
                "avg_context_text": " || ".join(raw_context_texts),
                "best_context_text": best_context_text,
                "topk_items": json.dumps(candidates),
            }
        )

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rerank SASRec top-k candidate lists with ContextRec using parquet user_prompt + generated_context."
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--llm_model", type=str, default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--user_profile_file", type=str, default=None)
    parser.add_argument(
        "--user_rep_source",
        type=str,
        default="parquet_user_prompt",
        choices=["parquet_user_prompt", "profile_embeddings"],
        help="Use per-case user_prompt from parquet by default. profile_embeddings keeps the old long-profile behavior.",
    )
    parser.add_argument("--sasrec_predictions_path", type=str, required=True)
    parser.add_argument("--context_path", action="append", default=[], help="Parquet context file(s). Can be passed multiple times.")
    parser.add_argument("--context_label", action="append", default=[], help="Optional labels aligned with --context_path.")
    parser.add_argument("--context_alpha", type=float, default=0.5)
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--parquet_user_id_col", type=str, default="auto", choices=["auto", "user_id", "original_user_id"])
    parser.add_argument("--parquet_item_id_col", type=str, default="auto", choices=["auto", "original_gt_item", "target_item", "item_id", "target_x"])
    parser.add_argument("--parquet_step_col", type=str, default="auto", choices=["auto", "step", "none"])
    parser.add_argument("--parquet_context_text_col", type=str, default="generated_context", choices=["auto", "generated_context"])
    parser.add_argument("--parquet_history_text_col", type=str, default="user_prompt", choices=["auto", "user_prompt", "none"])
    parser.add_argument("--text_mode", type=str, default="full", choices=["auto", "full", "situation"])
    parser.add_argument("--max_cases", type=int, default=None)
    parser.add_argument("--save_case_reports", action="store_true")
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
    print(f"Loading item embeddings from: {item_embeddings_path}")
    print(f"Loading SASRec predictions from: {args.sasrec_predictions_path}")

    user_embeddings_path = None
    user_df = None
    user_dict: Dict[int, np.ndarray] = {}
    if args.user_rep_source == "profile_embeddings":
        user_embeddings_path = Path(
            resolve_requested_or_latest_user_profile_embedding_file(
                data_dir=data_dir,
                model_name_short=args.llm_model.split("/")[-1],
                explicit_path=args.user_profile_file,
            )
        )
        print(f"Loading user embeddings from: {user_embeddings_path}")
        user_df = load_pickle(str(user_embeddings_path))
        user_dict = {int(row["user_id"]): np.asarray(row["profile"], dtype=np.float32) for _, row in user_df.iterrows()}
    else:
        print("Using parquet user_prompt as the user representation source.")

    item_df = load_pickle(str(item_embeddings_path))
    sasrec_df = pd.read_parquet(args.sasrec_predictions_path)

    required_cols = {"original_user_id", "step", "target_x", "topk_items", "target_rank", "target_probability"}
    missing_cols = sorted(required_cols - set(sasrec_df.columns))
    if missing_cols:
        raise KeyError(f"SASRec predictions file is missing columns: {missing_cols}")

    item_dict = {int(row["item_id"]): np.asarray(row["description"], dtype=np.float32) for _, row in item_df.iterrows()}
    print(f"Loaded {len(item_dict)} item embeddings")
    if user_dict:
        print(f"Loaded {len(user_dict)} user embeddings")

    model = RecSysModel(embed_dim=384, hidden_dim=128, dropout=0.2).to(device)
    model = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    context_inputs: List[Tuple[str, str]] = []
    for idx, path in enumerate(args.context_path):
        label = args.context_label[idx] if idx < len(args.context_label) else auto_label_for_path(path)
        context_inputs.append((path, label))

    if not context_inputs:
        raise ValueError("No context sources provided. Pass one or more parquet files with --context_path.")

    encoder = MeanPoolingTextEncoder(args.sbert_model, device=device)

    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if user_embeddings_path is not None:
        profile_tag_match = re.search(r"_N(\d+)_T(\d+)(?:\.pkl)$", user_embeddings_path.name)
        profile_tag = f"N{profile_tag_match.group(1)}_T{profile_tag_match.group(2)}" if profile_tag_match else user_embeddings_path.stem
    else:
        profile_tag = "userprompt"

    summary = OrderedDict()
    for path, label in context_inputs:
        print(f"\nLoading context source: {label} ({path})")
        source = load_context_source(
            path=path,
            label=label,
            encoder=encoder,
            encode_batch_size=args.encode_batch_size,
            parquet_user_id_col=args.parquet_user_id_col,
            parquet_item_id_col=args.parquet_item_id_col,
            parquet_step_col=args.parquet_step_col,
            parquet_context_text_col=args.parquet_context_text_col,
            parquet_history_text_col=args.parquet_history_text_col,
            text_mode=args.text_mode,
        )
        print(
            f"  pair contexts={source.pair_count} | case contexts={source.case_count} | total contexts={source.total_contexts} "
            f"| pair histories={len(source.pair_history_embeddings)} | case histories={len(source.case_history_embeddings)}"
        )

        case_report = rerank_source_against_sasrec(
            model=model,
            user_dict=user_dict,
            item_dict=item_dict,
            sasrec_df=sasrec_df,
            source=source,
            args=args,
            device=device,
            max_cases=args.max_cases,
        )
        print(f"  matched reranking cases={len(case_report)}")

        if case_report.empty:
            source_summary = {
                "path": path,
                "pair_count": source.pair_count,
                "case_count": source.case_count,
                "total_contexts": source.total_contexts,
                "matched_cases": 0,
                "user_col_used": source.user_col_used,
                "item_col_used": source.item_col_used,
                "step_col_used": source.step_col_used,
                "context_text_col_used": source.context_text_col_used,
                "history_text_col_used": source.history_text_col_used,
                "text_mode_used": source.text_mode_used,
            }
        else:
            sasrec_vs_long = summarize_case_report(case_report, "sasrec_candidate_rank", "contextrec_long_only_rank")
            sasrec_vs_avg = summarize_case_report(case_report, "sasrec_candidate_rank", "contextrec_avg_rank")
            sasrec_vs_best = summarize_case_report(case_report, "sasrec_candidate_rank", "contextrec_best_rank")

            print(
                f"  SASRec -> ContextRec long-only: avg rank {sasrec_vs_long['avg_rank_before']:.2f} -> {sasrec_vs_long['avg_rank_after']:.2f} "
                f"(avg delta {sasrec_vs_long['avg_delta']:.2f})"
            )
            print(
                f"  SASRec -> ContextRec avg-context: avg rank {sasrec_vs_avg['avg_rank_before']:.2f} -> {sasrec_vs_avg['avg_rank_after']:.2f} "
                f"(avg delta {sasrec_vs_avg['avg_delta']:.2f})"
            )
            print(
                f"  SASRec -> ContextRec best-context: avg rank {sasrec_vs_best['avg_rank_before']:.2f} -> {sasrec_vs_best['avg_rank_after']:.2f} "
                f"(avg delta {sasrec_vs_best['avg_delta']:.2f})"
            )

            source_summary = {
                "path": path,
                "pair_count": source.pair_count,
                "case_count": source.case_count,
                "total_contexts": source.total_contexts,
                "matched_cases": int(len(case_report)),
                "user_col_used": source.user_col_used,
                "item_col_used": source.item_col_used,
                "step_col_used": source.step_col_used,
                "context_text_col_used": source.context_text_col_used,
                "history_text_col_used": source.history_text_col_used,
                "text_mode_used": source.text_mode_used,
                "user_rep_source": args.user_rep_source,
                "sasrec_vs_contextrec_long_only": sasrec_vs_long,
                "sasrec_vs_contextrec_avg": sasrec_vs_avg,
                "sasrec_vs_contextrec_best": sasrec_vs_best,
            }

            if args.save_case_reports:
                report_path = results_dir / f"rerank_sasrec_{args.dataset}_{profile_tag}_{label}_{timestamp}_case_report.csv"
                case_report.to_csv(report_path, index=False)
                source_summary["case_report_csv"] = str(report_path)
                print(f"  Saved case report to {report_path}")

        summary[label] = source_summary

    out_path = results_dir / f"rerank_sasrec_{args.dataset}_{profile_tag}_{timestamp}.json"
    payload = {
        "dataset": args.dataset,
        "llm_model": args.llm_model,
        "user_rep_source": args.user_rep_source,
        "user_profile_file": str(user_embeddings_path) if user_embeddings_path is not None else None,
        "checkpoint_path": str(checkpoint_path),
        "sasrec_predictions_path": args.sasrec_predictions_path,
        "context_alpha": args.context_alpha,
        "execution_time_sec": time.time() - t0,
        "context_sources": summary,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved reranking summary to {out_path}")


if __name__ == "__main__":
    main()
