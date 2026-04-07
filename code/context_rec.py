import math
import time
import random
import pickle
import json
import re
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


DATASET = "movies"
SCRIPT_NAME = "context_rec.py"
METHOD_NAME = "ContextRec"


# -------------------
# Set seeds
# -------------------
def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Seed set to {seed}")
    else:
        print("No seed provided. Using default randomness.")


# -------------------
# Configuration
# -------------------
class Config:
    def __init__(self, llm_model="microsoft/phi-3-mini-4k-instruct"):
        self.llm_model = llm_model
        self.model_name_short = llm_model.split("/")[-1] if "/" in llm_model else llm_model

        root = Path(__file__).resolve().parents[1]
        data_dir = root / "data" / DATASET

        # Absolute paths so the script works no matter where it is launched from
        self.ITEM_EMBEDDINGS_PATH = str(data_dir / "bert_item_features.pkl")
        self.USER_LONG_TERM_PATH = None  # resolved in main()
        self.CONTEXT_PATH = str(data_dir / "bert_context_profiles_test.pkl")

        self.TRAIN_PATH = str(data_dir / "train.csv")
        self.VAL_PATH = str(data_dir / "validation.csv")
        self.TEST_PATH = str(data_dir / "test.csv")

        # Training hyperparameters
        self.BATCH_SIZE = 2048
        self.NUM_NEG_SAMPLES = 5
        self.LR = 1e-3
        self.WEIGHT_DECAY = 1e-5
        self.EPOCHS = 100
        self.EARLY_STOP_PATIENCE = 5

        # Model hyperparameters
        self.EMBEDDING_DIM = 384
        self.HIDDEN_DIM = 128
        self.DROPOUT = 0.2

        # Hardware settings
        self.NUM_WORKERS = 4
        self.MULTI_GPU = True

        # Context fusion: user_rep = alpha * user_long_emb + (1 - alpha) * context_emb
        self.CONTEXT_ALPHA = 0.5

        self.RUN_NAME = f"context_rec_{DATASET}"


# -------------------
# Utility functions
# -------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_interactions_csv(path):
    return pd.read_csv(path)


def save_run_results_json(
    script_name,
    method_name,
    dataset,
    seed,
    cfg,
    checkpoint_path,
    test_metrics,
    ranking_results_baseline,
    ranking_results_context_avg,
    ranking_results_context_best,
    execution_time_sec,
    profile_tag,
):
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_seed = "none" if seed is None else str(seed)
    json_path = results_dir / f"{Path(script_name).stem}_{dataset}_{profile_tag}_seed{safe_seed}_{timestamp}.json"

    payload = {
        "script_name": script_name,
        "method_name": method_name,
        "dataset": dataset,
        "seed": seed,
        "timestamp": timestamp,
        "profile_tag": profile_tag,
        "checkpoint_path": str(checkpoint_path),
        "config": {
            "ITEM_EMBEDDINGS_PATH": cfg.ITEM_EMBEDDINGS_PATH,
            "USER_LONG_TERM_PATH": cfg.USER_LONG_TERM_PATH,
            "CONTEXT_PATH": cfg.CONTEXT_PATH,
            "TRAIN_PATH": cfg.TRAIN_PATH,
            "VAL_PATH": cfg.VAL_PATH,
            "TEST_PATH": cfg.TEST_PATH,
            "BATCH_SIZE": cfg.BATCH_SIZE,
            "NUM_NEG_SAMPLES": cfg.NUM_NEG_SAMPLES,
            "LR": cfg.LR,
            "WEIGHT_DECAY": cfg.WEIGHT_DECAY,
            "EPOCHS": cfg.EPOCHS,
            "EARLY_STOP_PATIENCE": cfg.EARLY_STOP_PATIENCE,
            "EMBEDDING_DIM": cfg.EMBEDDING_DIM,
            "HIDDEN_DIM": cfg.HIDDEN_DIM,
            "DROPOUT": cfg.DROPOUT,
            "NUM_WORKERS": cfg.NUM_WORKERS,
            "MULTI_GPU": cfg.MULTI_GPU,
            "CONTEXT_ALPHA": cfg.CONTEXT_ALPHA,
            "RUN_NAME": cfg.RUN_NAME,
            "llm_model": cfg.llm_model,
            "model_name_short": cfg.model_name_short,
        },
        "test_metrics": test_metrics,
        "ranking_results_baseline": ranking_results_baseline,
        "ranking_results_context_avg": ranking_results_context_avg,
        "ranking_results_context_best": ranking_results_context_best,
        "execution_time_sec": execution_time_sec,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved run results to {json_path}")


def negative_sampling(positive_item_ids, all_item_ids, num_neg_samples):
    candidate_items = list(all_item_ids - set(positive_item_ids))
    if not candidate_items:
        return []
    if len(candidate_items) <= num_neg_samples:
        return candidate_items
    return random.sample(candidate_items, num_neg_samples)


def resolve_user_profile_embedding_file(data_dir: Path, model_name_short: str) -> str:
    """
    Find the long-profile EMBEDDING file, excluding *_text.pkl.
    Prefer the most recently modified matching file.
    """
    candidates = sorted(
        p
        for p in data_dir.glob(f"bert_long_term_user_profiles_{model_name_short}_N*_T*.pkl")
        if not p.name.endswith("_text.pkl")
    )

    if candidates:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(chosen)

    fallback = data_dir / f"bert_long_term_user_profiles_{model_name_short}.pkl"
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        f"No long-profile embedding files found for model {model_name_short} in {data_dir}\n"
        f"Expected something like:\n"
        f"  bert_long_term_user_profiles_{model_name_short}_N*_T*.pkl\n"
        f"Run: python preprocessing/generate_profiles.py --llm_model <model>"
    )




def resolve_requested_or_latest_user_profile_embedding_file(
    data_dir: Path,
    model_name_short: str,
    explicit_path: str | None,
) -> str:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_absolute():
            candidate = (data_dir / explicit_path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Requested user profile embedding file not found: {explicit_path}")
        if candidate.name.endswith("_text.pkl"):
            raise ValueError(
                "--user_profile_file must point to the embedding pickle, not the *_text.pkl file. "
                f"Got: {candidate.name}"
            )
        return str(candidate)

    return resolve_user_profile_embedding_file(data_dir, model_name_short)


def extract_profile_tag(profile_path: str) -> str:
    name = Path(profile_path).name
    match = re.search(r"_N(\d+)_T(\d+)(?:\.pkl)$", name)
    if match:
        return f"N{match.group(1)}_T{match.group(2)}"

    stem = Path(profile_path).stem
    prefix = "bert_long_term_user_profiles_"
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-_")
    return stem or "profile"


def print_split_coverage(split_name, df, available_users, available_items):
    split_users = set(df["user_id"].astype(int).unique())
    split_items = set(df["item_id"].astype(int).unique())
    missing_users = sorted(split_users - available_users)
    missing_items = sorted(split_items - available_items)

    print(
        f"{split_name}: {len(df)} rows | "
        f"{len(split_users)} users ({len(missing_users)} missing) | "
        f"{len(split_items)} items ({len(missing_items)} missing)"
    )
    if missing_users:
        print(f"  First missing users: {missing_users[:10]}")
    if missing_items:
        print(f"  First missing items: {missing_items[:10]}")


def filter_split(df, available_users, available_items, split_name):
    original_len = len(df)
    filtered = df[
        df["user_id"].isin(available_users) &
        df["item_id"].isin(available_items)
    ].copy()

    print(
        f"{split_name}: kept {len(filtered)}/{original_len} rows "
        f"after filtering to users/items with available embeddings"
    )
    return filtered


# -------------------
# Dataset
# -------------------
class RecSysDataset(Dataset):
    """
    Dataset for training. Uses only the long-term user profile embedding.
    No context embeddings are used during training.
    """

    def __init__(self, interactions_df, user_dict, item_dict, num_neg_samples=5):
        self.interactions = interactions_df.reset_index(drop=True)
        self.user_dict = user_dict
        self.item_dict = item_dict
        self.num_neg_samples = num_neg_samples

        self.user_pos_item_map = {}
        for row in self.interactions.itertuples(index=False):
            u_id = int(getattr(row, "user_id"))
            i_id = int(getattr(row, "item_id"))
            self.user_pos_item_map.setdefault(u_id, set()).add(i_id)

        self.users = list(self.interactions["user_id"].astype(int))
        self.items = list(self.interactions["item_id"].astype(int))
        self.all_item_ids = set(self.item_dict.keys())

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        u_id = self.users[idx]
        pos_i_id = self.items[idx]

        user_emb = self.user_dict[u_id]
        pos_item_emb = self.item_dict[pos_i_id]

        data = [(
            np.array(user_emb, dtype=np.float32),
            pos_item_emb.astype(np.float32),
            1
        )]

        neg_item_ids = negative_sampling(
            positive_item_ids=self.user_pos_item_map[u_id],
            all_item_ids=self.all_item_ids,
            num_neg_samples=self.num_neg_samples
        )

        for neg_i in neg_item_ids:
            neg_item_emb = self.item_dict[neg_i]
            data.append((
                np.array(user_emb, dtype=np.float32),
                neg_item_emb.astype(np.float32),
                0
            ))

        return data


# -------------------
# Collate Function
# -------------------
def collate_fn(batch):
    flat_data = [pair for sample in batch for pair in sample]
    user_embs, item_embs, labels = zip(*flat_data)
    return (
        torch.stack([torch.from_numpy(u) for u in user_embs]),
        torch.stack([torch.from_numpy(i) for i in item_embs]),
        torch.tensor(labels, dtype=torch.long),
    )


# -------------------
# Model
# -------------------
class RecSysModel(nn.Module):
    """
    Simple MLP recommender trained on long-term user profiles.
    Input: [user_emb (384) | item_emb (384)] -> 2 logits.
    At inference, user_emb can be replaced by a context-fused representation
    of the same dimensionality without any architectural changes.
    """

    def __init__(self, embed_dim=384, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self._initialize_weights()

    def forward(self, user_emb, item_emb):
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


# -------------------
# Evaluation Metrics
# -------------------
def accuracy(preds, labels):
    predicted = torch.argmax(preds, dim=1)
    return (predicted == labels).sum().item() / labels.size(0)


def precision_recall_ndcg(preds, labels):
    predicted = torch.argmax(preds, dim=1)
    tp = ((predicted == 1) & (labels == 1)).sum().item()
    fp = ((predicted == 1) & (labels == 0)).sum().item()
    fn = ((predicted == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision, recall, precision  # ndcg placeholder


def dcg_at_k(relevance_list, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_list[:k]))


def ndcg_at_k(relevance_list, k):
    dcg = dcg_at_k(relevance_list, k)
    idcg = dcg_at_k(sorted(relevance_list, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def _score_user(model, user_rep_torch, item_embs_torch, device):
    user_rep_expanded = user_rep_torch.repeat(item_embs_torch.size(0), 1)
    concat_vec = torch.cat([user_rep_expanded, item_embs_torch], dim=1)
    with torch.no_grad():
        fc = model.module.fc if isinstance(model, nn.DataParallel) else model.fc
        logits = fc(concat_vec)
    return torch.softmax(logits, dim=1)[:, 1]


def _ranking_metrics(ranked_item_ids, relevant_items, top_k_list):
    num_relevant = len(relevant_items)
    results = {}
    for k in top_k_list:
        top_k = ranked_item_ids[:k]
        hits = sum(1 for i in top_k if i in relevant_items)
        results[k] = {
            "precision": hits / k,
            "recall": hits / num_relevant if num_relevant > 0 else 0.0,
            "ndcg": ndcg_at_k([1 if i in relevant_items else 0 for i in top_k], k),
        }
    return results


# -------------------
# Ranking Evaluation
# -------------------
def evaluate_ranking(model, user_dict, item_dict, test_df, device, top_k_list=(10, 20)):
    model.eval()

    user_to_test_items = {}
    for row in test_df.itertuples(index=False):
        u_id = int(getattr(row, "user_id"))
        i_id = int(getattr(row, "item_id"))
        user_to_test_items.setdefault(u_id, set()).add(i_id)

    all_item_ids = list(item_dict.keys())
    item_embs_torch = torch.tensor(
        [item_dict[i].astype("float32") for i in all_item_ids]
    ).to(device)

    acc = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}
    n = 0

    for user_id, test_items in user_to_test_items.items():
        if user_id not in user_dict:
            continue

        user_torch = torch.tensor(
            np.array(user_dict[user_id], dtype=np.float32)
        ).unsqueeze(0).to(device)

        probs = _score_user(model, user_torch, item_embs_torch, device)
        ranked = [all_item_ids[i] for i in torch.argsort(probs, descending=True).tolist()]

        for k, m in _ranking_metrics(ranked, test_items, top_k_list).items():
            for metric, val in m.items():
                acc[k][metric] += val
        n += 1

    if n == 0:
        return {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}

    return {k: {m: v / n for m, v in acc[k].items()} for k in top_k_list}


def evaluate_ranking_with_context(
    model, user_dict, item_dict, test_df, context_dict, alpha, device, top_k_list=(10, 20)
):
    model.eval()

    user_to_test_items = {}
    for row in test_df.itertuples(index=False):
        u_id = int(getattr(row, "user_id"))
        i_id = int(getattr(row, "item_id"))
        user_to_test_items.setdefault(u_id, set()).add(i_id)

    all_item_ids = list(item_dict.keys())
    item_embs_torch = torch.tensor(
        [item_dict[i].astype("float32") for i in all_item_ids]
    ).to(device)

    acc_avg = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}
    acc_best = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}
    n = 0

    for user_id, test_items in user_to_test_items.items():
        if user_id not in user_dict:
            continue

        user_long_np = np.array(user_dict[user_id], dtype=np.float32)

        user_contexts = {
            item_id: context_dict.get((user_id, item_id), [])
            for item_id in test_items
        }

        if not any(user_contexts.values()):
            continue

        # avg fusion
        all_ctx_embs = [emb for ctxs in user_contexts.values() for emb in ctxs]
        if all_ctx_embs:
            mean_ctx = np.mean(all_ctx_embs, axis=0).astype(np.float32)
            fused_avg = alpha * user_long_np + (1 - alpha) * mean_ctx
            user_torch_avg = torch.tensor(fused_avg).unsqueeze(0).to(device)
            probs_avg = _score_user(model, user_torch_avg, item_embs_torch, device)
            ranked_avg = [all_item_ids[i] for i in torch.argsort(probs_avg, descending=True).tolist()]
            for k, m in _ranking_metrics(ranked_avg, test_items, top_k_list).items():
                for metric, val in m.items():
                    acc_avg[k][metric] += val

        # best-of-K fusion (oracle)
        best_per_k = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}
        has_best = False

        for item_id, ctxs in user_contexts.items():
            relevant_items = {item_id}
            for ctx_emb in ctxs:
                fused = alpha * user_long_np + (1 - alpha) * ctx_emb.astype(np.float32)
                user_torch_ctx = torch.tensor(fused).unsqueeze(0).to(device)
                probs_ctx = _score_user(model, user_torch_ctx, item_embs_torch, device)
                ranked_ctx = [all_item_ids[i] for i in torch.argsort(probs_ctx, descending=True).tolist()]
                metrics_ctx = _ranking_metrics(ranked_ctx, relevant_items, top_k_list)

                for k in top_k_list:
                    for metric in ("precision", "recall", "ndcg"):
                        if metrics_ctx[k][metric] > best_per_k[k][metric]:
                            best_per_k[k][metric] = metrics_ctx[k][metric]
                has_best = True

        if has_best:
            for k in top_k_list:
                for metric in ("precision", "recall", "ndcg"):
                    acc_best[k][metric] += best_per_k[k][metric]

        n += 1

    if n == 0:
        empty = {k: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0} for k in top_k_list}
        return {"avg": empty, "best": empty}

    return {
        "avg": {k: {m: v / n for m, v in acc_avg[k].items()} for k in top_k_list},
        "best": {k: {m: v / n for m, v in acc_best[k].items()} for k in top_k_list},
    }


# -------------------
# Training and Validation
# -------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    print("Starting Training...")
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0

    for batch_idx, (user_emb, item_emb, labels) in enumerate(train_loader):
        print(f"Training Batch: {batch_idx}")
        user_emb, item_emb, labels = user_emb.to(device), item_emb.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(user_emb, item_emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += accuracy(logits, labels)

    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    print("Start Validation...")
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_prec = 0.0
    val_rec = 0.0
    val_ndcg = 0.0

    with torch.no_grad():
        for batch_idx, (user_emb, item_emb, labels) in enumerate(val_loader):
            print(f"Validation Batch: {batch_idx}")
            user_emb, item_emb, labels = user_emb.to(device), item_emb.to(device), labels.to(device)

            logits = model(user_emb, item_emb)
            loss = criterion(logits, labels)

            val_loss += loss.item()
            val_acc += accuracy(logits, labels)
            p, r, n = precision_recall_ndcg(logits, labels)
            val_prec += p
            val_rec += r
            val_ndcg += n

    n_batches = len(val_loader)
    return (
        val_loss / n_batches,
        val_acc / n_batches,
        val_prec / n_batches,
        val_rec / n_batches,
        val_ndcg / n_batches,
    )


# -------------------
# Early Stopping
# -------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=1e-4, checkpoint_path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


# -------------------
# Main
# -------------------
def main(seed, llm_model="microsoft/phi-3-mini-4k-instruct"):
    cfg = Config(llm_model=llm_model)
    run_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using LLM model: {llm_model}")

    data_dir = Path(__file__).resolve().parents[1] / "data" / DATASET
    cfg.USER_LONG_TERM_PATH = resolve_user_profile_embedding_file(data_dir, cfg.model_name_short)

    print(f"Loading user profiles from: {Path(cfg.USER_LONG_TERM_PATH).name}")

    # -------------------
    # Data Loading
    # -------------------
    print("Loading data...")
    item_df = load_pickle(cfg.ITEM_EMBEDDINGS_PATH)
    user_df = load_pickle(cfg.USER_LONG_TERM_PATH)

    train_df = load_interactions_csv(cfg.TRAIN_PATH)
    val_df = load_interactions_csv(cfg.VAL_PATH)
    test_df = load_interactions_csv(cfg.TEST_PATH)

    item_dict = {int(row["item_id"]): row["description"] for _, row in item_df.iterrows()}
    user_dict = {int(row["user_id"]): row["profile"] for _, row in user_df.iterrows()}

    available_users = set(user_dict.keys())
    available_items = set(item_dict.keys())

    print("\nCoverage before filtering")
    print("-------------------------")
    print_split_coverage("train", train_df, available_users, available_items)
    print_split_coverage("validation", val_df, available_users, available_items)
    print_split_coverage("test", test_df, available_users, available_items)

    train_df = filter_split(train_df, available_users, available_items, "train")
    val_df = filter_split(val_df, available_users, available_items, "validation")
    test_df = filter_split(test_df, available_users, available_items, "test")

    if len(train_df) == 0:
        raise RuntimeError("Train split is empty after filtering. Generate more user profiles or use matching filtered CSVs.")
    if len(val_df) == 0:
        raise RuntimeError("Validation split is empty after filtering. Generate more user profiles or use matching filtered CSVs.")
    if len(test_df) == 0:
        raise RuntimeError("Test split is empty after filtering. Generate more user profiles or use matching filtered CSVs.")

    print("\nCoverage after filtering")
    print("------------------------")
    print(f"train rows: {len(train_df)}")
    print(f"validation rows: {len(val_df)}")
    print(f"test rows: {len(test_df)}")
    print(f"available user embeddings: {len(user_dict)}")
    print(f"available item embeddings: {len(item_dict)}")

    # -------------------
    # Datasets & Loaders
    # -------------------
    train_dataset = RecSysDataset(train_df, user_dict, item_dict, cfg.NUM_NEG_SAMPLES)
    val_dataset = RecSysDataset(val_df, user_dict, item_dict, cfg.NUM_NEG_SAMPLES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # -------------------
    # Model, Optimizer, Criterion
    # -------------------
    model = RecSysModel(cfg.EMBEDDING_DIM, cfg.HIDDEN_DIM, cfg.DROPOUT)

    if cfg.MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # -------------------
    # Training Loop
    # -------------------
    checkpoint_path = f"best_{Path(SCRIPT_NAME).stem}_{DATASET}_seed_{seed}.pt"
    early_stopper = EarlyStopping(
        patience=cfg.EARLY_STOP_PATIENCE,
        verbose=True,
        checkpoint_path=checkpoint_path,
    )

    for epoch in range(cfg.EPOCHS):
        print(f"Processing Epoch {epoch + 1} ...")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_ndcg = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{cfg.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val NDCG: {val_ndcg:.4f}"
        )

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # -------------------
    # Load best model
    # -------------------
    print("Loading best model weights...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # -------------------
    # Batch-level Testing
    # -------------------
    test_dataset = RecSysDataset(test_df, user_dict, item_dict, cfg.NUM_NEG_SAMPLES)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print("Starting Testing...")
    test_loss, test_acc, test_prec, test_rec, test_ndcg = evaluate(model, test_loader, criterion, device)
    print(
        f"Test Results -> Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, "
        f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, NDCG: {test_ndcg:.4f}"
    )

    # -------------------
    # Ranking - Baseline (no context)
    # -------------------
    print("Starting Ranking-Based Testing (Baseline)...")
    top_k_list = [10, 20]
    ranking_baseline = evaluate_ranking(
        model=model,
        user_dict=user_dict,
        item_dict=item_dict,
        test_df=test_df,
        device=device,
        top_k_list=top_k_list,
    )
    for k in top_k_list:
        print(
            f"[Baseline] Top-{k} => "
            f"Precision: {ranking_baseline[k]['precision']:.4f}, "
            f"Recall: {ranking_baseline[k]['recall']:.4f}, "
            f"NDCG: {ranking_baseline[k]['ndcg']:.4f}"
        )

    # -------------------
    # Ranking - With Context
    # -------------------
    print("Starting Ranking-Based Testing (With Context)...")
    context_path = Path(cfg.CONTEXT_PATH)
    ranking_ctx_avg = None
    ranking_ctx_best = None

    if context_path.exists():
        context_dict = load_pickle(str(context_path))

        # Optional visibility into context coverage on the filtered test split
        filtered_test_pairs = {(int(r.user_id), int(r.item_id)) for r in test_df.itertuples(index=False)}
        covered_pairs = sum(1 for pair in filtered_test_pairs if pair in context_dict)
        print(f"Context coverage on filtered test pairs: {covered_pairs}/{len(filtered_test_pairs)}")

        context_results = evaluate_ranking_with_context(
            model=model,
            user_dict=user_dict,
            item_dict=item_dict,
            test_df=test_df,
            context_dict=context_dict,
            alpha=cfg.CONTEXT_ALPHA,
            device=device,
            top_k_list=top_k_list,
        )
        ranking_ctx_avg = context_results["avg"]
        ranking_ctx_best = context_results["best"]

        for k in top_k_list:
            print(
                f"[Context-Avg]  Top-{k} => "
                f"Precision: {ranking_ctx_avg[k]['precision']:.4f}, "
                f"Recall: {ranking_ctx_avg[k]['recall']:.4f}, "
                f"NDCG: {ranking_ctx_avg[k]['ndcg']:.4f}"
            )
            print(
                f"[Context-Best] Top-{k} => "
                f"Precision: {ranking_ctx_best[k]['precision']:.4f}, "
                f"Recall: {ranking_ctx_best[k]['recall']:.4f}, "
                f"NDCG: {ranking_ctx_best[k]['ndcg']:.4f}"
            )
    else:
        print(
            f"Context file not found at {context_path}. "
            f"Run generate_contexts.py first to produce context embeddings."
        )

    # -------------------
    # Save results
    # -------------------
    test_metrics = {
        "loss": test_loss,
        "accuracy": test_acc,
        "precision": test_prec,
        "recall": test_rec,
        "ndcg": test_ndcg,
    }

    save_run_results_json(
        SCRIPT_NAME,
        METHOD_NAME,
        DATASET,
        seed,
        cfg,
        checkpoint_path,
        test_metrics,
        ranking_baseline,
        ranking_ctx_avg,
        ranking_ctx_best,
        time.time() - run_start_time,
        profile_tag,
    )
    print("Done.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ContextRec: LLM-generated GT-aligned contexts for recommendation."
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument(
        "--llm_model",
        type=str,
        default="microsoft/phi-3-mini-4k-instruct",
        help="LLM model used to generate profiles (e.g. 'microsoft/phi-3-mini-4k-instruct')",
    )
    parser.add_argument(
        "--user_profile_file",
        type=str,
        default=None,
        help=(
            "Embedding pickle to use for long-term user descriptions, either absolute or relative to data/<dataset>. "
            "Example: bert_long_term_user_profiles_phi-3-mini-4k-instruct_N5000_T015.pkl"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(f"ContextRec-{DATASET}")
    args = parse_arguments()
    set_seed(args.seed)
    main(args.seed, args.llm_model, args.user_profile_file)
    print(f"ContextRec-{DATASET} Done")