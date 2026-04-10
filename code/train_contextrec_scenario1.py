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


SCRIPT_NAME = "train_contextrec_scenario1.py"
METHOD_NAME = "ContextRecScenario1"

CaseKey = Tuple[int, int, int]


@dataclass
class Scenario1Case:
    original_user_id: int
    step: int
    target_item: int
    user_prompt: str
    generated_context: str
    topk_items: List[int]
    target_rank: Optional[int]
    target_probability: Optional[float]


@dataclass
class EncodedCase:
    original_user_id: int
    step: int
    target_item: int
    user_prompt: str
    generated_context: str
    user_emb: np.ndarray
    context_emb: np.ndarray
    candidates: List[int]
    baseline_rank: int
    full_rank: Optional[int]
    target_probability: Optional[float]


class MeanPoolingTextEncoder:
    def __init__(self, model_name: str, device: torch.device):
        resolved_name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
        self.model_name = resolved_name
        self.device = device
        print(f"Loading text encoder: {resolved_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        self.model = AutoModel.from_pretrained(resolved_name).to(device)
        self.model.eval()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 256,
        max_length: int = 256,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        outputs: List[np.ndarray] = []
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


class FusionMLPRecSysModel(nn.Module):
    """
    Minimal history+context+item scorer.

    History and context are fused through a small learned gate:
        gate = sigmoid(W [history ; context])
        fused = gate * history + (1 - gate) * context

    Final prediction is an MLP over [fused ; item].
    """

    def __init__(self, embed_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self._initialize_weights()

    def fuse(self, history_emb: torch.Tensor, context_emb: torch.Tensor) -> torch.Tensor:
        gate = self.fusion_gate(torch.cat([history_emb, context_emb], dim=1))
        return gate * history_emb + (1.0 - gate) * context_emb

    def forward(self, history_emb: torch.Tensor, context_emb: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        fused = self.fuse(history_emb, context_emb)
        return self.scorer(torch.cat([fused, item_emb], dim=1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class Scenario1PairDataset(Dataset):
    def __init__(self, cases: Sequence[EncodedCase], item_dict: Dict[int, np.ndarray], neg_per_case: int):
        self.cases = list(cases)
        self.item_dict = item_dict
        self.neg_per_case = neg_per_case

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        samples = [
            (
                case.user_emb.astype(np.float32),
                case.context_emb.astype(np.float32),
                self.item_dict[case.target_item].astype(np.float32),
                1,
            )
        ]

        negatives = [item_id for item_id in case.candidates if item_id != case.target_item and item_id in self.item_dict]
        if negatives:
            sample_count = min(self.neg_per_case, len(negatives))
            for item_id in random.sample(negatives, sample_count):
                samples.append(
                    (
                        case.user_emb.astype(np.float32),
                        case.context_emb.astype(np.float32),
                        self.item_dict[item_id].astype(np.float32),
                        0,
                    )
                )
        return samples


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
    return pq.read_table(path).to_pandas()


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


def clean_text(text: str) -> Optional[str]:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    if cleaned.upper() == "NO_CONTEXT":
        return None
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def encode_unique_texts(texts: Sequence[str], encoder: MeanPoolingTextEncoder, batch_size: int) -> Dict[str, np.ndarray]:
    unique_texts = list(dict.fromkeys(texts))
    embs = encoder.encode(unique_texts, batch_size=batch_size)
    return {text: emb.astype(np.float32) for text, emb in zip(unique_texts, embs)}


def collate_pairs(batch):
    flat = [pair for sample in batch for pair in sample]
    history_embs, context_embs, item_embs, labels = zip(*flat)
    return (
        torch.tensor(np.stack(history_embs), dtype=torch.float32),
        torch.tensor(np.stack(context_embs), dtype=torch.float32),
        torch.tensor(np.stack(item_embs), dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for history_emb, context_emb, item_emb, labels in loader:
        history_emb = history_emb.to(device)
        context_emb = context_emb.to(device)
        item_emb = item_emb.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(history_emb, context_emb, item_emb)
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
        for history_emb, context_emb, item_emb, labels in loader:
            history_emb = history_emb.to(device)
            context_emb = context_emb.to(device)
            item_emb = item_emb.to(device)
            labels = labels.to(device)

            logits = model(history_emb, context_emb, item_emb)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            total_acc += (preds == labels).float().mean().item()

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


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


def score_candidates(
    model: FusionMLPRecSysModel,
    history_rep: np.ndarray,
    context_rep: np.ndarray,
    candidates: List[int],
    item_dict: Dict[int, np.ndarray],
    device: torch.device,
) -> Dict[int, float]:
    if not candidates:
        return {}
    history_tensor = torch.tensor(history_rep, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(candidates), 1)
    context_tensor = torch.tensor(context_rep, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(candidates), 1)
    item_tensor = torch.tensor(np.stack([item_dict[i].astype(np.float32) for i in candidates]), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(history_tensor, context_tensor, item_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return {item_id: float(score) for item_id, score in zip(candidates, probs)}


def evaluate_reranking(model, cases: Sequence[EncodedCase], item_dict: Dict[int, np.ndarray], device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for case in cases:
        candidates = [item_id for item_id in case.candidates if item_id in item_dict]
        if case.target_item not in candidates:
            candidates = list(candidates) + [case.target_item]
        if case.target_item not in candidates:
            continue

        scores = score_candidates(model, case.user_emb, case.context_emb, candidates, item_dict, device)
        ranked = sorted(candidates, key=lambda item_id: scores[item_id], reverse=True)
        rerank = rank_of_item(ranked, case.target_item)
        rows.append(
            {
                "original_user_id": case.original_user_id,
                "step": case.step,
                "target_item": case.target_item,
                "user_prompt": case.user_prompt,
                "generated_context": case.generated_context,
                "num_candidates": len(candidates),
                "sasrec_candidate_rank": case.baseline_rank,
                "sasrec_full_rank": case.full_rank,
                "sasrec_target_probability": case.target_probability,
                "contextrec_rank": rerank,
                "delta_vs_sasrec": case.baseline_rank - rerank,
            }
        )
    return pd.DataFrame(rows)


def split_cases(cases: Sequence[EncodedCase], seed: int, train_frac: float, val_frac: float):
    users = sorted({case.original_user_id for case in cases})
    rnd = random.Random(seed)
    rnd.shuffle(users)

    n_users = len(users)
    n_train = max(1, int(round(n_users * train_frac)))
    n_val = max(1, int(round(n_users * val_frac)))
    n_train = min(n_train, n_users - 2) if n_users >= 3 else min(n_train, n_users)
    n_val = min(n_val, n_users - n_train - 1) if n_users - n_train >= 2 else max(0, n_users - n_train - 1)

    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train + n_val])
    test_users = set(users[n_train + n_val:])

    if not test_users:
        if val_users:
            moved = next(iter(val_users))
            val_users.remove(moved)
            test_users.add(moved)
        elif train_users and len(train_users) > 1:
            moved = next(iter(train_users))
            train_users.remove(moved)
            test_users.add(moved)

    train_cases = [case for case in cases if case.original_user_id in train_users]
    val_cases = [case for case in cases if case.original_user_id in val_users]
    test_cases = [case for case in cases if case.original_user_id in test_users]
    return train_cases, val_cases, test_cases


def load_scenario1_cases(context_path: str, sasrec_path: str) -> List[Scenario1Case]:
    ctx_df = load_parquet_df(context_path)
    required_ctx = {"original_user_id", "step", "target_item", "user_prompt", "generated_context"}
    missing_ctx = sorted(required_ctx - set(ctx_df.columns))
    if missing_ctx:
        raise KeyError(f"Scenario 1 parquet is missing required columns: {missing_ctx}")

    sasrec_df = load_parquet_df(sasrec_path)
    required_sasrec = {"original_user_id", "step", "target_x", "topk_items", "target_rank", "target_probability"}
    missing_sasrec = sorted(required_sasrec - set(sasrec_df.columns))
    if missing_sasrec:
        raise KeyError(f"SASRec parquet is missing required columns: {missing_sasrec}")

    ctx = ctx_df[["original_user_id", "step", "target_item", "user_prompt", "generated_context"]].copy()
    ctx["original_user_id"] = ctx["original_user_id"].astype(int)
    ctx["step"] = ctx["step"].astype(int)
    ctx["target_item"] = ctx["target_item"].astype(int)
    ctx["user_prompt"] = ctx["user_prompt"].astype(str)
    ctx["generated_context"] = ctx["generated_context"].map(clean_text)
    ctx = ctx.dropna(subset=["generated_context"]).drop_duplicates().reset_index(drop=True)

    sasrec = sasrec_df[["original_user_id", "step", "target_x", "topk_items", "target_rank", "target_probability"]].copy()
    sasrec["original_user_id"] = sasrec["original_user_id"].astype(int)
    sasrec["step"] = sasrec["step"].astype(int)
    sasrec["target_x"] = sasrec["target_x"].astype(int)
    sasrec["topk_items"] = sasrec["topk_items"].map(parse_listlike)

    merged = ctx.merge(sasrec, how="left", on=["original_user_id", "step"], validate="m:1")
    merged = merged[merged["target_x"].notna()].copy()
    merged["target_x"] = merged["target_x"].astype(int)
    merged = merged[merged["target_item"] == merged["target_x"]].copy()

    if merged.empty:
        raise RuntimeError(
            "No rows matched on original_user_id + step + target item. "
            "Please check that the Scenario 1 parquet and SASRec parquet use the same item ids."
        )

    cases: List[Scenario1Case] = []
    for row in merged.itertuples(index=False):
        topk_items = [int(i) for i in row.topk_items]
        cases.append(
            Scenario1Case(
                original_user_id=int(row.original_user_id),
                step=int(row.step),
                target_item=int(row.target_item),
                user_prompt=str(row.user_prompt),
                generated_context=str(row.generated_context),
                topk_items=topk_items,
                target_rank=None if pd.isna(row.target_rank) else int(row.target_rank),
                target_probability=None if pd.isna(row.target_probability) else float(row.target_probability),
            )
        )
    return cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Scenario-1-only MovieLens reranker on user_prompt + generated_context with negatives sampled from SASRec topk items."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sbert_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--item_embeddings_path", type=str, required=True)
    parser.add_argument("--sasrec_predictions_path", type=str, required=True)
    parser.add_argument("--scenario1_context_path", type=str, required=True)
    parser.add_argument("--encode_batch_size", type=int, default=256)
    parser.add_argument("--neg_per_case", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--max_cases", type=int, default=None)
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

    print(f"Loading item embeddings from: {args.item_embeddings_path}")
    item_df = load_pickle(args.item_embeddings_path)
    item_dict = {int(row["item_id"]): np.asarray(row["description"], dtype=np.float32) for _, row in item_df.iterrows()}
    print(f"Loaded {len(item_dict)} item embeddings")

    raw_cases = load_scenario1_cases(args.scenario1_context_path, args.sasrec_predictions_path)
    if args.max_cases is not None:
        raw_cases = raw_cases[:args.max_cases]
    print(f"Loaded {len(raw_cases)} Scenario 1 matched rows")
    if not raw_cases:
        raise RuntimeError("No Scenario 1 cases available after matching the parquet files.")

    encoder = MeanPoolingTextEncoder(args.sbert_model, device=device)
    user_emb_map = encode_unique_texts([case.user_prompt for case in raw_cases], encoder, args.encode_batch_size)
    context_emb_map = encode_unique_texts([case.generated_context for case in raw_cases], encoder, args.encode_batch_size)

    encoded_cases: List[EncodedCase] = []
    skipped_missing_target = 0
    for case in raw_cases:
        if case.target_item not in item_dict:
            skipped_missing_target += 1
            continue
        candidates = [item_id for item_id in case.topk_items if item_id in item_dict]
        if case.target_item not in candidates:
            candidates = list(candidates) + [case.target_item]
        if len(candidates) < 2:
            continue
        baseline_rank = candidates.index(case.target_item) + 1
        encoded_cases.append(
            EncodedCase(
                original_user_id=case.original_user_id,
                step=case.step,
                target_item=case.target_item,
                user_prompt=case.user_prompt,
                generated_context=case.generated_context,
                user_emb=user_emb_map[case.user_prompt],
                context_emb=context_emb_map[case.generated_context],
                candidates=candidates,
                baseline_rank=baseline_rank,
                full_rank=case.target_rank,
                target_probability=case.target_probability,
            )
        )
    print(
        f"Prepared {len(encoded_cases)} encoded Scenario 1 cases "
        f"({skipped_missing_target} skipped because the target item embedding was missing)"
    )
    if len(encoded_cases) < 3:
        raise RuntimeError("Need at least 3 encoded cases to build train/val/test splits.")

    train_cases, val_cases, test_cases = split_cases(encoded_cases, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac)
    print(
        f"Split by original_user_id -> train={len(train_cases)} val={len(val_cases)} test={len(test_cases)}"
    )
    if not train_cases or not val_cases or not test_cases:
        raise RuntimeError("One of the splits is empty. Try lowering train_frac/val_frac or use more cases.")

    train_loader = DataLoader(
        Scenario1PairDataset(train_cases, item_dict, neg_per_case=args.neg_per_case),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
    )
    val_loader = DataLoader(
        Scenario1PairDataset(val_cases, item_dict, neg_per_case=args.neg_per_case),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_pairs,
    )

    model = FusionMLPRecSysModel(embed_dim=384, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = script_dir / f"best_contextrec_scenario1_seed{args.seed}_{timestamp}.pt"
    stopper = EarlyStopping(patience=args.patience, checkpoint_path=str(checkpoint_path))

    train_history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_pairs(model, val_loader, criterion, device)
        train_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        print(
            f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        stopper(val_loss, model)
        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_report = evaluate_reranking(model, test_cases, item_dict, device)
    rerank_summary = summarize_rank_report(test_report, "sasrec_candidate_rank", "contextrec_rank")
    print(
        f"Scenario 1 reranking: avg rank {rerank_summary['avg_rank_before']:.2f} -> "
        f"{rerank_summary['avg_rank_after']:.2f} (avg delta {rerank_summary['avg_delta']:.2f})"
    )

    case_report_path = None
    if args.save_case_report:
        case_report_path = results_dir / f"contextrec_scenario1_seed{args.seed}_{timestamp}_case_report.csv"
        test_report.to_csv(case_report_path, index=False)
        print(f"Saved case report to {case_report_path}")

    summary = {
        "script_name": SCRIPT_NAME,
        "method_name": METHOD_NAME,
        "seed": args.seed,
        "timestamp": timestamp,
        "sbert_model": args.sbert_model,
        "item_embeddings_path": args.item_embeddings_path,
        "sasrec_predictions_path": args.sasrec_predictions_path,
        "scenario1_context_path": args.scenario1_context_path,
        "checkpoint_path": str(checkpoint_path),
        "case_report_path": None if case_report_path is None else str(case_report_path),
        "num_item_embeddings": len(item_dict),
        "num_total_cases": len(encoded_cases),
        "num_train_users": len({case.original_user_id for case in train_cases}),
        "num_val_users": len({case.original_user_id for case in val_cases}),
        "num_test_users": len({case.original_user_id for case in test_cases}),
        "num_train_cases": len(train_cases),
        "num_val_cases": len(val_cases),
        "num_test_cases": len(test_cases),
        "neg_per_case": args.neg_per_case,
        "train_history": train_history,
        "test_rerank": rerank_summary,
        "execution_time_sec": time.time() - t0,
    }
    summary_path = results_dir / f"contextrec_scenario1_seed{args.seed}_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
