import argparse
import ast
import json
import math
import pickle
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from context_rec import RecSysModel


DATASET = "movielens"
SCRIPT_NAME = "train_contextrec_from_parquet.py"
METHOD_NAME = "ContextRecMovieLensParquet"

CaseKey = Tuple[int, int, int]


@dataclass
class BaseTrainCase:
    original_user_id: int
    step: int
    target_item: int
    user_emb: np.ndarray
    candidates: List[int]
    baseline_rank: int
    full_rank: Optional[int]
    target_probability: Optional[float]


@dataclass
class ContextEvalCase:
    source: str
    original_user_id: int
    step: int
    target_item: int
    user_prompt: str
    contexts: List[str]
    user_emb: np.ndarray
    context_embs: List[np.ndarray]
    candidates: List[int]
    baseline_rank: int
    full_rank: Optional[int]
    target_probability: Optional[float]


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
            processed = min(start + batch_size, len(texts))
            if (start // batch_size + 1) % 25 == 0 or processed == len(texts):
                print(f"  Encoded {processed}/{len(texts)} texts")
        return np.concatenate(outputs, axis=0)


class PairwiseCaseDataset(Dataset):
    def __init__(self, cases: Sequence[BaseTrainCase], item_dict: Dict[int, np.ndarray], neg_per_case: int):
        self.cases = list(cases)
        self.item_dict = item_dict
        self.neg_per_case = neg_per_case

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        pos_item = self.item_dict[case.target_item]
        examples = [(case.user_emb.astype(np.float32), pos_item.astype(np.float32), 1)]
        negatives = [i for i in case.candidates if i != case.target_item and i in self.item_dict]
        if not negatives:
            return examples
        sample_count = min(self.neg_per_case, len(negatives))
        sampled_negatives = random.sample(negatives, sample_count)
        for item_id in sampled_negatives:
            examples.append((case.user_emb.astype(np.float32), self.item_dict[item_id].astype(np.float32), 0))
        return examples


class EarlyStopping:
    def __init__(self, patience: int, checkpoint_path: str, delta: float = 1e-4):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_val_loss = math.inf
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.best_val_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            return
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True


SITUATION_RE = re.compile(r"SITUATION:\s*(.*?)(?:\n+\s*(?:TRIGGER|HISTORY_LINK):|\Z)", re.IGNORECASE | re.DOTALL)


def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_parquet_df(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


def parse_listlike(value) -> List[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    text = str(value).strip()
    if not text or text == "nan":
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, (list, tuple, np.ndarray)):
        return [int(x) for x in parsed]
    return []


def clean_context_text(text: str, mode: str = "full") -> Optional[str]:
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


def collate_pairs(batch):
    flat = [pair for sample in batch for pair in sample]
    user_embs, item_embs, labels = zip(*flat)
    return (
        torch.tensor(np.stack(user_embs), dtype=torch.float32),
        torch.tensor(np.stack(item_embs), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for user_emb, item_emb, labels in loader:
        user_emb = user_emb.to(device)
        item_emb = item_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(user_emb, item_emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_acc += (preds == labels).float().mean().item()
    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


def evaluate_pairs(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for user_emb, item_emb, labels in loader:
            user_emb = user_emb.to(device)
            item_emb = item_emb.to(device)
            labels = labels.to(device)
            logits = model(user_emb, item_emb)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            total_acc += (preds == labels).float().mean().item()
    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


def score_candidates(model, user_rep: np.ndarray, candidates: List[int], item_dict: Dict[int, np.ndarray], device: torch.device) -> Dict[int, float]:
    item_tensor = torch.tensor(np.stack([item_dict[i].astype(np.float32) for i in candidates]), dtype=torch.float32, device=device)
    user_tensor = torch.tensor(user_rep.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(0).repeat(len(candidates), 1)
    with torch.no_grad():
        logits = model(user_tensor, item_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {item_id: float(score) for item_id, score in zip(candidates, probs)}


def rank_of_item(ranked_items: Sequence[int], target_item: int) -> int:
    return ranked_items.index(target_item) + 1


def build_hit_metrics(df: pd.DataFrame, rank_col: str, ks=(1, 3, 5, 10)) -> Dict[str, float]:
    if df.empty:
        return {f"hit@{k}": 0.0 for k in ks}
    return {f"hit@{k}": float((df[rank_col] <= k).mean()) for k in ks}


def summarize_rank_report(df: pd.DataFrame, before_col: str, after_col: str) -> Dict[str, float]:
    if df.empty:
        return {
            "case_count": 0,
            "avg_rank_before": 0.0,
            "avg_rank_after": 0.0,
            "avg_delta": 0.0,
            "improved_count": 0,
            "unchanged_count": 0,
            "worsened_count": 0,
            **{f"before_hit@{k}": 0.0 for k in (1, 3, 5, 10)},
            **{f"after_hit@{k}": 0.0 for k in (1, 3, 5, 10)},
        }
    delta = df[before_col] - df[after_col]
    summary = {
        "case_count": int(len(df)),
        "avg_rank_before": float(df[before_col].mean()),
        "avg_rank_after": float(df[after_col].mean()),
        "avg_delta": float(delta.mean()),
        "improved_count": int((delta > 0).sum()),
        "unchanged_count": int((delta == 0).sum()),
        "worsened_count": int((delta < 0).sum()),
    }
    summary.update({f"before_{k}": v for k, v in build_hit_metrics(df, before_col).items()})
    summary.update({f"after_{k}": v for k, v in build_hit_metrics(df, after_col).items()})
    return summary


def normalize_item_embeddings(item_frame: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, str]]:
    required = {"item_id", "description"}
    missing = sorted(required - set(item_frame.columns))
    if missing:
        raise KeyError(f"Item embedding pickle is missing required columns: {missing}")
    item_dict: Dict[int, np.ndarray] = {}
    history_lookup: Dict[int, str] = {}
    brief_col = "history_brief" if "history_brief" in item_frame.columns else None
    title_col = "title" if "title" in item_frame.columns else None
    for row in item_frame.itertuples(index=False):
        item_id = int(getattr(row, "item_id"))
        emb = np.asarray(getattr(row, "description"), dtype=np.float32)
        item_dict[item_id] = emb
        if brief_col is not None:
            history_lookup[item_id] = str(getattr(row, brief_col))
        elif title_col is not None:
            history_lookup[item_id] = str(getattr(row, title_col))
        else:
            history_lookup[item_id] = f"Item {item_id}"
    return item_dict, history_lookup


def render_history_text(item_ids: Sequence[int], history_lookup: Dict[int, str], max_items: int) -> str:
    filtered = [int(i) for i in item_ids if int(i) in history_lookup]
    if max_items is not None and max_items > 0:
        filtered = filtered[-max_items:]
    lines = [
        "A user of a movie streaming platform watched the following movies in this order (oldest to newest):",
        "",
    ]
    for idx, item_id in enumerate(filtered, start=1):
        lines.append(f"{idx}. {history_lookup[item_id]}")
    if len(lines) == 2:
        lines.append("1. [no known history]")
    return "\n".join(lines)


def split_users(all_users: Sequence[int], seed: int, train_frac: float, val_frac: float):
    users = list(sorted(set(int(u) for u in all_users)))
    rnd = random.Random(seed)
    rnd.shuffle(users)
    n = len(users)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train + n_val])
    test_users = set(users[n_train + n_val:])
    if not test_users:
        moved = next(iter(val_users))
        val_users.remove(moved)
        test_users.add(moved)
    return train_users, val_users, test_users


def load_sasrec_rows(sasrec_path: str) -> pd.DataFrame:
    df = load_parquet_df(sasrec_path)
    required = {"original_user_id", "step", "target_x", "input_history", "topk_items", "target_rank", "target_probability"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"SASRec parquet is missing required columns: {missing}")
    out = df[list(required)].copy()
    out["original_user_id"] = out["original_user_id"].astype(int)
    out["step"] = out["step"].astype(int)
    out["target_x"] = out["target_x"].astype(int)
    out["input_history"] = out["input_history"].map(parse_listlike)
    out["topk_items"] = out["topk_items"].map(parse_listlike)
    return out


def build_base_cases(
    df: pd.DataFrame,
    item_dict: Dict[int, np.ndarray],
    history_lookup: Dict[int, str],
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
    history_items: int,
) -> List[BaseTrainCase]:
    history_texts = [render_history_text(items, history_lookup, max_items=history_items) for items in df["input_history"].tolist()]
    history_embs = encoder.encode(history_texts, batch_size=encode_batch_size)

    cases: List[BaseTrainCase] = []
    skipped_missing_target = 0
    skipped_short_candidate = 0
    for row, user_emb in zip(df.itertuples(index=False), history_embs):
        target_item = int(row.target_x)
        if target_item not in item_dict:
            skipped_missing_target += 1
            continue
        candidates = [int(i) for i in row.topk_items if int(i) in item_dict]
        if target_item not in candidates:
            candidates.append(target_item)
        if len(candidates) < 2:
            skipped_short_candidate += 1
            continue
        baseline_rank = candidates.index(target_item) + 1
        cases.append(
            BaseTrainCase(
                original_user_id=int(row.original_user_id),
                step=int(row.step),
                target_item=target_item,
                user_emb=user_emb.astype(np.float32),
                candidates=candidates,
                baseline_rank=baseline_rank,
                full_rank=None if pd.isna(row.target_rank) else int(row.target_rank),
                target_probability=None if pd.isna(row.target_probability) else float(row.target_probability),
            )
        )
    print(
        f"Prepared {len(cases)} base training/eval cases "
        f"({skipped_missing_target} skipped missing target embedding, {skipped_short_candidate} skipped short candidate lists)"
    )
    return cases


def choose_context_columns(df: pd.DataFrame) -> Tuple[str, str, str, str, str]:
    user_col = "original_user_id" if "original_user_id" in df.columns else "user_id"
    if "original_gt_item" in df.columns:
        item_col = "original_gt_item"
    elif "target_item" in df.columns:
        item_col = "target_item"
    elif "target_x" in df.columns:
        item_col = "target_x"
    else:
        raise KeyError(f"No usable item column found in {df.columns.tolist()}")
    if "step" not in df.columns:
        raise KeyError("Expected a 'step' column in parquet contexts")
    if "user_prompt" not in df.columns:
        raise KeyError("Expected a 'user_prompt' column in parquet contexts")
    if "generated_context" not in df.columns:
        raise KeyError("Expected a 'generated_context' column in parquet contexts")
    return user_col, item_col, "step", "user_prompt", "generated_context"


def standardize_context_df(df: pd.DataFrame, label: str, text_mode: str) -> pd.DataFrame:
    user_col, item_col, step_col, history_col, context_col = choose_context_columns(df)
    print(
        f"Selected context columns for {label}: user='{user_col}' item='{item_col}' step='{step_col}' "
        f"history='{history_col}' context='{context_col}'"
    )
    out = df[[user_col, step_col, item_col, history_col, context_col]].copy()
    out.columns = ["original_user_id", "step", "target_item", "user_prompt", "generated_context"]
    out["source"] = label
    out["original_user_id"] = out["original_user_id"].astype(int)
    out["step"] = out["step"].astype(int)
    out["target_item"] = out["target_item"].astype(int)
    out["user_prompt"] = out["user_prompt"].astype(str)
    out["generated_context"] = out["generated_context"].map(lambda x: clean_context_text(x, mode=text_mode))
    out = out.dropna(subset=["generated_context"]).reset_index(drop=True)
    return out


def load_context_eval_cases(
    context_inputs: List[Tuple[str, str]],
    sasrec_df: pd.DataFrame,
    test_users: set,
    item_dict: Dict[int, np.ndarray],
    encoder: MeanPoolingTextEncoder,
    encode_batch_size: int,
    text_mode: str,
    max_eval_cases: Optional[int],
) -> List[ContextEvalCase]:
    sasrec_merge = sasrec_df[["original_user_id", "step", "target_x", "topk_items", "target_rank", "target_probability"]].copy()

    all_context_rows = []
    for path, label in context_inputs:
        print(f"Loading context parquet: {label} ({path})")
        df = load_parquet_df(path)
        std = standardize_context_df(df, label=label, text_mode=text_mode)
        print(f"  kept {len(std)}/{len(df)} rows after dropping empty/NO_CONTEXT contexts")
        all_context_rows.append(std)

    if not all_context_rows:
        return []

    merged_contexts = pd.concat(all_context_rows, ignore_index=True)
    merged_contexts = merged_contexts[merged_contexts["original_user_id"].isin(test_users)].copy()
    print(f"Context rows on held-out users: {len(merged_contexts)}")

    merged = merged_contexts.merge(sasrec_merge, how="left", on=["original_user_id", "step"], validate="m:1")
    merged = merged[merged["target_x"].notna()].copy()
    merged["target_x"] = merged["target_x"].astype(int)
    merged = merged[merged["target_item"] == merged["target_x"]].copy()
    print(f"Exact held-out context/SASRec matches: {len(merged)}")

    grouped: Dict[CaseKey, Dict[str, object]] = {}
    for row in merged.itertuples(index=False):
        key = (int(row.original_user_id), int(row.step), int(row.target_item))
        bucket = grouped.setdefault(
            key,
            {
                "source": str(row.source),
                "user_prompt": str(row.user_prompt),
                "contexts": [],
                "candidates": [int(i) for i in row.topk_items if int(i) in item_dict],
                "target_rank": None if pd.isna(row.target_rank) else int(row.target_rank),
                "target_probability": None if pd.isna(row.target_probability) else float(row.target_probability),
            },
        )
        bucket["contexts"].append(str(row.generated_context))

    grouped_items = list(grouped.items())
    if max_eval_cases is not None:
        grouped_items = grouped_items[:max_eval_cases]

    user_prompt_texts = [str(payload["user_prompt"]) for _, payload in grouped_items]
    unique_user_prompts = list(dict.fromkeys(user_prompt_texts))
    user_prompt_embs = encoder.encode(unique_user_prompts, batch_size=encode_batch_size)
    user_prompt_map = {text: emb.astype(np.float32) for text, emb in zip(unique_user_prompts, user_prompt_embs)}

    context_texts = []
    for _, payload in grouped_items:
        context_texts.extend(dedupe_preserve_order(payload["contexts"]))
    unique_context_texts = list(dict.fromkeys(context_texts))
    context_embs = encoder.encode(unique_context_texts, batch_size=encode_batch_size)
    context_map = {text: emb.astype(np.float32) for text, emb in zip(unique_context_texts, context_embs)}

    cases: List[ContextEvalCase] = []
    for (original_user_id, step, target_item), payload in grouped_items:
        contexts = dedupe_preserve_order(payload["contexts"])
        candidates = list(payload["candidates"])
        if target_item not in candidates:
            candidates.append(target_item)
        if len(candidates) < 2:
            continue
        cases.append(
            ContextEvalCase(
                source=str(payload["source"]),
                original_user_id=int(original_user_id),
                step=int(step),
                target_item=int(target_item),
                user_prompt=str(payload["user_prompt"]),
                contexts=contexts,
                user_emb=user_prompt_map[str(payload["user_prompt"])],
                context_embs=[context_map[text] for text in contexts if text in context_map],
                candidates=candidates,
                baseline_rank=candidates.index(target_item) + 1,
                full_rank=payload["target_rank"],
                target_probability=payload["target_probability"],
            )
        )
    print(f"Prepared {len(cases)} held-out context evaluation cases")
    return cases


def evaluate_reranking(model, cases: Sequence[ContextEvalCase], item_dict: Dict[int, np.ndarray], alpha: float, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for case in cases:
        candidates = [i for i in case.candidates if i in item_dict]
        if case.target_item not in candidates:
            candidates.append(case.target_item)
        if case.target_item not in candidates:
            continue

        long_scores = score_candidates(model, case.user_emb, candidates, item_dict, device)
        long_order = sorted(candidates, key=lambda item_id: long_scores[item_id], reverse=True)
        long_rank = rank_of_item(long_order, case.target_item)

        avg_rank = None
        best_rank = None
        best_context_text = None
        avg_context_text = None
        if case.context_embs:
            mean_ctx = np.mean(np.stack(case.context_embs, axis=0), axis=0).astype(np.float32)
            fused_avg = alpha * case.user_emb + (1 - alpha) * mean_ctx
            avg_scores = score_candidates(model, fused_avg, candidates, item_dict, device)
            avg_order = sorted(candidates, key=lambda item_id: avg_scores[item_id], reverse=True)
            avg_rank = rank_of_item(avg_order, case.target_item)
            avg_context_text = " || ".join(case.contexts)

            best_target_score = -np.inf
            for ctx_text, ctx_emb in zip(case.contexts, case.context_embs):
                fused = alpha * case.user_emb + (1 - alpha) * ctx_emb
                ctx_scores = score_candidates(model, fused, candidates, item_dict, device)
                ctx_order = sorted(candidates, key=lambda item_id: ctx_scores[item_id], reverse=True)
                ctx_rank = rank_of_item(ctx_order, case.target_item)
                target_score = ctx_scores[case.target_item]
                if best_rank is None or ctx_rank < best_rank or (ctx_rank == best_rank and target_score > best_target_score):
                    best_rank = ctx_rank
                    best_target_score = target_score
                    best_context_text = ctx_text

        rows.append(
            {
                "source": case.source,
                "original_user_id": case.original_user_id,
                "step": case.step,
                "target_item": case.target_item,
                "user_prompt": case.user_prompt,
                "num_candidates": len(candidates),
                "num_contexts": len(case.context_embs),
                "topk_plus_gt": json.dumps(candidates),
                "sasrec_candidate_rank": case.baseline_rank,
                "sasrec_full_rank": case.full_rank,
                "sasrec_target_probability": case.target_probability,
                "contextrec_long_only_rank": long_rank,
                "contextrec_avg_rank": avg_rank,
                "contextrec_best_rank": best_rank,
                "delta_long_only_vs_sasrec": case.baseline_rank - long_rank,
                "delta_avg_vs_sasrec": None if avg_rank is None else case.baseline_rank - avg_rank,
                "delta_best_vs_sasrec": None if best_rank is None else case.baseline_rank - best_rank,
                "avg_context_text": avg_context_text,
                "best_context_text": best_context_text,
            }
        )
    return pd.DataFrame(rows)


def build_source_summaries(test_report: pd.DataFrame) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for source, source_df in test_report.groupby("source"):
        out[str(source)] = {
            "long_only": summarize_rank_report(source_df, "sasrec_candidate_rank", "contextrec_long_only_rank"),
            "avg_context": summarize_rank_report(source_df.dropna(subset=["contextrec_avg_rank"]), "sasrec_candidate_rank", "contextrec_avg_rank"),
            "best_context": summarize_rank_report(source_df.dropna(subset=["contextrec_best_rank"]), "sasrec_candidate_rank", "contextrec_best_rank"),
        }
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Train a MovieLens reranker from SASRec parquet data and evaluate context reranking on held-out MovieLens context parquets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--item_embeddings_path", type=str, required=True)
    parser.add_argument("--sasrec_predictions_path", type=str, required=True)
    parser.add_argument("--context_path", action="append", default=[], required=True)
    parser.add_argument("--context_label", action="append", default=[])
    parser.add_argument("--text_mode", type=str, default="full", choices=["full", "auto", "situation"])
    parser.add_argument("--history_items", type=int, default=20)
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--neg_per_case", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--context_alpha", type=float, default=0.5)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--max_eval_cases", type=int, default=None)
    parser.add_argument("--save_case_report", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    item_frame = load_pickle(args.item_embeddings_path)
    if not isinstance(item_frame, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame in {args.item_embeddings_path}, got {type(item_frame)}")
    item_dict, history_lookup = normalize_item_embeddings(item_frame)
    print(f"Loaded {len(item_dict)} item embeddings from {args.item_embeddings_path}")

    sasrec_df = load_sasrec_rows(args.sasrec_predictions_path)
    print(f"Loaded {len(sasrec_df)} SASRec rows from {args.sasrec_predictions_path}")

    train_users, val_users, test_users = split_users(sasrec_df["original_user_id"].tolist(), seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
    print(f"User split -> train={len(train_users)} val={len(val_users)} test={len(test_users)}")

    train_df = sasrec_df[sasrec_df["original_user_id"].isin(train_users)].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_df = sasrec_df[sasrec_df["original_user_id"].isin(val_users)].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.max_train_rows is not None:
        train_df = train_df.head(args.max_train_rows).copy()
    if args.max_val_rows is not None:
        val_df = val_df.head(args.max_val_rows).copy()
    print(f"Base train rows={len(train_df)} val rows={len(val_df)}")

    encoder = MeanPoolingTextEncoder(args.sbert_model, device=device)
    train_cases = build_base_cases(train_df, item_dict, history_lookup, encoder, args.encode_batch_size, args.history_items)
    val_cases = build_base_cases(val_df, item_dict, history_lookup, encoder, args.encode_batch_size, args.history_items)
    if not train_cases or not val_cases:
        raise RuntimeError("Training or validation cases are empty after preprocessing.")

    context_inputs = []
    for idx, path in enumerate(args.context_path):
        label = args.context_label[idx] if idx < len(args.context_label) else Path(path).stem
        context_inputs.append((path, label))
    eval_cases = load_context_eval_cases(
        context_inputs=context_inputs,
        sasrec_df=sasrec_df,
        test_users=test_users,
        item_dict=item_dict,
        encoder=encoder,
        encode_batch_size=args.encode_batch_size,
        text_mode=args.text_mode,
        max_eval_cases=args.max_eval_cases,
    )
    if not eval_cases:
        raise RuntimeError("No held-out context evaluation cases were matched. Check the MovieLens parquet files and split sizes.")

    train_loader = DataLoader(PairwiseCaseDataset(train_cases, item_dict, neg_per_case=args.neg_per_case), batch_size=args.batch_size, shuffle=True, collate_fn=collate_pairs)
    val_loader = DataLoader(PairwiseCaseDataset(val_cases, item_dict, neg_per_case=args.neg_per_case), batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)

    model = RecSysModel(embed_dim=384, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = "-".join(label for _, label in context_inputs)
    source_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", source_tag).strip("-_") or "contexts"
    checkpoint_path = script_dir / f"best_contextrec_movielens_{source_tag}_hist{args.history_items}_seed{args.seed}_{timestamp}.pt"

    stopper = EarlyStopping(patience=args.patience, checkpoint_path=str(checkpoint_path))
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_pairs(model, val_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        print(
            f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        stopper(val_loss, model)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_report = evaluate_reranking(model, eval_cases, item_dict, alpha=args.context_alpha, device=device)

    long_summary = summarize_rank_report(test_report, "sasrec_candidate_rank", "contextrec_long_only_rank")
    avg_summary = summarize_rank_report(test_report.dropna(subset=["contextrec_avg_rank"]), "sasrec_candidate_rank", "contextrec_avg_rank")
    best_summary = summarize_rank_report(test_report.dropna(subset=["contextrec_best_rank"]), "sasrec_candidate_rank", "contextrec_best_rank")
    source_summaries = build_source_summaries(test_report)

    print(
        f"Held-out reranking (long-only): avg rank {long_summary['avg_rank_before']:.2f} -> {long_summary['avg_rank_after']:.2f} "
        f"(avg delta {long_summary['avg_delta']:.2f})"
    )
    print(
        f"Held-out reranking (avg context): avg rank {avg_summary['avg_rank_before']:.2f} -> {avg_summary['avg_rank_after']:.2f} "
        f"(avg delta {avg_summary['avg_delta']:.2f})"
    )
    print(
        f"Held-out reranking (best context): avg rank {best_summary['avg_rank_before']:.2f} -> {best_summary['avg_rank_after']:.2f} "
        f"(avg delta {best_summary['avg_delta']:.2f})"
    )

    case_report_path = None
    if args.save_case_report:
        case_report_path = results_dir / f"contextrec_movielens_{source_tag}_hist{args.history_items}_seed{args.seed}_{timestamp}_case_report.csv"
        test_report.to_csv(case_report_path, index=False)
        print(f"Saved case report to {case_report_path}")

    summary = {
        "script_name": SCRIPT_NAME,
        "method_name": METHOD_NAME,
        "seed": args.seed,
        "timestamp": timestamp,
        "sbert_model": args.sbert_model,
        "item_embeddings_path": str(args.item_embeddings_path),
        "sasrec_predictions_path": args.sasrec_predictions_path,
        "context_sources": [label for _, label in context_inputs],
        "history_items": args.history_items,
        "text_mode": args.text_mode,
        "checkpoint_path": str(checkpoint_path),
        "case_report_path": None if case_report_path is None else str(case_report_path),
        "num_item_embeddings": len(item_dict),
        "num_train_users": len(train_users),
        "num_val_users": len(val_users),
        "num_test_users": len(test_users),
        "num_train_rows": len(train_df),
        "num_val_rows": len(val_df),
        "num_train_cases": len(train_cases),
        "num_val_cases": len(val_cases),
        "num_eval_cases": len(eval_cases),
        "train_history": history,
        "heldout_rerank_long_only": long_summary,
        "heldout_rerank_avg_context": avg_summary,
        "heldout_rerank_best_context": best_summary,
        "per_source_heldout_summaries": source_summaries,
        "execution_time_sec": time.time() - t0,
    }
    summary_path = results_dir / f"contextrec_movielens_{source_tag}_hist{args.history_items}_seed{args.seed}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
