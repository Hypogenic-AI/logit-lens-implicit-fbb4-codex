#!/usr/bin/env python
"""Run an implicit-token logit-lens experiment on real LLM hidden states.

The experiment constructs a candidate implicit vocabulary from multi-token
WikiText-2 words, collects hidden states from a HuggingFace causal LM at the
last subtoken of each occurrence, and evaluates a tuned-lens-style ridge
translator whose output space is whole-word prototypes rather than tokenizer
IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{3,}")


@dataclass
class Config:
    seed: int = 42
    model_name: str = "Qwen/Qwen2.5-0.5B"
    dataset_path: str = "datasets/wikitext2_raw"
    output_dir: str = "results"
    figure_dir: str = "figures"
    hf_home: str = "artifacts/hf_cache"
    max_words: int = 128
    min_occurrences: int = 16
    train_per_word: int = 10
    val_per_word: int = 2
    test_per_word: int = 4
    final_group_min_types: int = 8
    max_words_per_final_group: int = 8
    max_context_chars: int = 420
    max_length: int = 96
    batch_size: int = 64
    ridge_lambda: float = 10.0
    bootstrap_samples: int = 3000
    randomization_samples: int = 6000
    dtype: str = "float16"
    device: str = "auto"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def normalize_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def clean_word(raw: str) -> str:
    return raw.strip("'-").lower()


def encode_word(tokenizer: Any, word: str) -> list[int]:
    return tokenizer.encode(" " + word, add_special_tokens=False)


def extract_all_occurrences(dataset_path: Path) -> list[dict[str, Any]]:
    dataset = load_from_disk(str(dataset_path))
    occurrences: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for row_idx, row in enumerate(dataset[split]):
            text = row["text"]
            if not text or not text.strip() or text.strip().startswith("="):
                continue
            for match in WORD_RE.finditer(text):
                word = clean_word(match.group(0))
                if len(word) < 5 or not word.isalpha():
                    continue
                start, end = match.span()
                left = max(0, end - 420)
                context = text[left:end]
                if len(context.strip()) < len(word):
                    continue
                occurrences.append(
                    {
                        "word": word,
                        "source_split": split,
                        "row_idx": row_idx,
                        "char_start": start,
                        "char_end": end,
                        "context": context,
                    }
                )
    return occurrences


def choose_candidate_words(
    occurrences: list[dict[str, Any]], tokenizer: Any, cfg: Config
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    counts = Counter(item["word"] for item in occurrences)
    word_meta: dict[str, dict[str, Any]] = {}
    groups: dict[int, list[str]] = defaultdict(list)

    for word, count in counts.items():
        if count < cfg.min_occurrences:
            continue
        piece_ids = encode_word(tokenizer, word)
        if len(piece_ids) < 2:
            continue
        final_id = piece_ids[-1]
        word_meta[word] = {
            "word": word,
            "count": int(count),
            "piece_ids": piece_ids,
            "pieces": tokenizer.convert_ids_to_tokens(piece_ids),
            "final_token_id": int(final_id),
            "final_token": tokenizer.convert_ids_to_tokens([final_id])[0],
        }
        groups[final_id].append(word)

    eligible_groups = [
        words for words in groups.values() if len(words) >= cfg.final_group_min_types
    ]
    eligible_groups.sort(key=lambda ws: (-sum(counts[w] for w in ws), min(ws)))

    chosen: list[str] = []
    for words in eligible_groups:
        words = sorted(words, key=lambda w: (-counts[w], w))
        for word in words[: cfg.max_words_per_final_group]:
            chosen.append(word)
            if len(chosen) >= cfg.max_words:
                break
        if len(chosen) >= cfg.max_words:
            break

    if len(chosen) < cfg.max_words:
        already = set(chosen)
        leftovers = sorted(
            (w for w in word_meta if w not in already), key=lambda w: (-counts[w], w)
        )
        chosen.extend(leftovers[: cfg.max_words - len(chosen)])

    chosen_set = set(chosen)
    selected_occurrences = [item for item in occurrences if item["word"] in chosen_set]
    selected_meta = {word: word_meta[word] for word in chosen}
    return selected_occurrences, selected_meta


def make_balanced_split(
    occurrences: list[dict[str, Any]], word_meta: dict[str, dict[str, Any]], cfg: Config
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(cfg.seed)
    by_word: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in occurrences:
        by_word[item["word"]].append(item)

    examples: list[dict[str, Any]] = []
    kept_words: list[str] = []
    need = cfg.train_per_word + cfg.val_per_word + cfg.test_per_word
    for label, word in enumerate(word_meta.keys()):
        items = by_word[word][:]
        if len(items) < need:
            continue
        rng.shuffle(items)
        split_items = (
            [("train", item) for item in items[: cfg.train_per_word]]
            + [
                ("validation", item)
                for item in items[cfg.train_per_word : cfg.train_per_word + cfg.val_per_word]
            ]
            + [
                ("test", item)
                for item in items[
                    cfg.train_per_word
                    + cfg.val_per_word : cfg.train_per_word
                    + cfg.val_per_word
                    + cfg.test_per_word
                ]
            ]
        )
        kept_words.append(word)
        for split, item in split_items:
            meta = word_meta[word]
            examples.append(
                {
                    "example_id": len(examples),
                    "word": word,
                    "label": label,
                    "split": split,
                    "context": item["context"][-cfg.max_context_chars :],
                    "source_split": item["source_split"],
                    "row_idx": item["row_idx"],
                    "piece_ids": meta["piece_ids"],
                    "pieces": meta["pieces"],
                    "final_token_id": meta["final_token_id"],
                    "final_token": meta["final_token"],
                }
            )

    relabel = {word: i for i, word in enumerate(kept_words)}
    final_meta = []
    for word in kept_words:
        meta = dict(word_meta[word])
        meta["label"] = relabel[word]
        final_meta.append(meta)
    for ex in examples:
        ex["label"] = relabel[ex["word"]]
    examples.sort(key=lambda item: item["example_id"])
    return examples, final_meta


def load_model_and_tokenizer(cfg: Config) -> tuple[Any, Any, torch.device]:
    os.environ.setdefault("HF_HOME", cfg.hf_home)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    dtype = torch.float16 if cfg.dtype == "float16" and device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, dtype=dtype, trust_remote_code=True
    )
    model.eval().to(device)
    return tokenizer, model, device


def get_final_norm(model: Any) -> Any | None:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    return None


def collect_hidden_states(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    cfg: Config,
) -> np.ndarray:
    norm_layer = get_final_norm(model)
    all_batches: list[np.ndarray] = []
    n = len(examples)
    started = time.time()
    for start in range(0, n, cfg.batch_size):
        batch = examples[start : start + cfg.batch_size]
        texts = [item["context"] for item in batch]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            add_special_tokens=False,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        last_positions = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            layer_vectors = []
            for hidden in outputs.hidden_states:
                vec = hidden[batch_indices, last_positions]
                if norm_layer is not None:
                    vec = norm_layer(vec)
                vec = normalize_torch(vec.float())
                layer_vectors.append(vec)
            stacked = torch.stack(layer_vectors, dim=1).detach().cpu().numpy()
        all_batches.append(stacked.astype(np.float16))
        done = min(start + cfg.batch_size, n)
        if done == n or done % (cfg.batch_size * 5) == 0:
            elapsed = time.time() - started
            print(f"hidden states: {done}/{n} examples in {elapsed:.1f}s", flush=True)
    return np.concatenate(all_batches, axis=0)


def candidate_embedding_tables(
    final_meta: list[dict[str, Any]], model: Any
) -> tuple[np.ndarray, np.ndarray]:
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    mean_rows = []
    final_token_rows = []
    for meta in final_meta:
        ids = torch.tensor(meta["piece_ids"], dtype=torch.long)
        mean_rows.append(emb[ids].mean(dim=0).numpy())
        final_token_rows.append(emb[int(meta["final_token_id"])].numpy())
    return (
        normalize_np(np.stack(mean_rows, axis=0).astype(np.float32)),
        normalize_np(np.stack(final_token_rows, axis=0).astype(np.float32)),
    )


def compute_prototypes(
    hidden_states: np.ndarray, labels: np.ndarray, split: np.ndarray, n_classes: int
) -> np.ndarray:
    train_mask = split == "train"
    final_states = hidden_states[:, -1, :].astype(np.float32)
    prototypes = np.zeros((n_classes, final_states.shape[-1]), dtype=np.float32)
    for label in range(n_classes):
        rows = final_states[train_mask & (labels == label)]
        if len(rows) == 0:
            raise ValueError(f"No train examples for label {label}")
        prototypes[label] = normalize_np(rows).mean(axis=0)
    return normalize_np(prototypes)


def compute_layer_prototypes(
    hidden_states: np.ndarray, labels: np.ndarray, split: np.ndarray, n_classes: int
) -> np.ndarray:
    """Build one implicit vocabulary table per layer from training centroids."""
    train_mask = split == "train"
    n_layers = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    prototypes = np.zeros((n_layers, n_classes, hidden_size), dtype=np.float32)
    for layer in range(n_layers):
        states = hidden_states[:, layer, :].astype(np.float32)
        for label in range(n_classes):
            rows = states[train_mask & (labels == label)]
            if len(rows) == 0:
                raise ValueError(f"No train examples for label {label}")
            prototypes[layer, label] = normalize_np(rows).mean(axis=0)
        prototypes[layer] = normalize_np(prototypes[layer])
    return prototypes


def ranks_from_scores(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    correct = scores[np.arange(len(labels)), labels]
    return 1 + np.sum(scores > correct[:, None], axis=1)


def metrics_from_scores(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    ranks = ranks_from_scores(scores, labels)
    preds = np.argmax(scores, axis=1)
    return {
        "top1": float(np.mean(preds == labels)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "preds": preds,
        "ranks": ranks,
    }


def train_ridge_translator(
    x_train: np.ndarray,
    y_train: np.ndarray,
    ridge_lambda: float,
    device: torch.device,
) -> np.ndarray:
    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)
    ones = torch.ones((x.shape[0], 1), dtype=torch.float32, device=device)
    x_aug = torch.cat([x, ones], dim=1)
    xtx = x_aug.T @ x_aug
    reg = torch.eye(xtx.shape[0], dtype=torch.float32, device=device) * ridge_lambda
    reg[-1, -1] = 0.0
    xty = x_aug.T @ y
    weights = torch.linalg.solve(xtx + reg, xty)
    return weights.detach().cpu().numpy().astype(np.float32)


def apply_translator(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ones = np.ones((x.shape[0], 1), dtype=np.float32)
    x_aug = np.concatenate([x.astype(np.float32), ones], axis=1)
    return normalize_np(x_aug @ weights)


def evaluate_layer_scores(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
    prototypes: np.ndarray,
    layer_prototypes: np.ndarray,
    mean_embeddings: np.ndarray,
    final_token_embeddings: np.ndarray,
    final_meta: list[dict[str, Any]],
    cfg: Config,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, np.ndarray]]]]:
    n_layers = hidden_states.shape[1]
    n_classes = prototypes.shape[0]
    train_mask = split == "train"
    val_mask = split == "validation"
    test_mask = split == "test"
    target_proto_train = prototypes[labels[train_mask]]

    split_masks = {"validation": val_mask, "test": test_mask}
    rows: list[dict[str, Any]] = []
    detailed: dict[str, dict[int, dict[str, np.ndarray]]] = defaultdict(dict)

    final_token_ids = np.array([meta["final_token_id"] for meta in final_meta])
    unique_final_ids = sorted(set(int(x) for x in final_token_ids))
    final_id_to_col = {tok_id: i for i, tok_id in enumerate(unique_final_ids)}
    final_token_cols = np.array([final_id_to_col[int(x)] for x in final_token_ids])

    for layer in range(n_layers):
        x_all = hidden_states[:, layer, :].astype(np.float32)
        x_train = x_all[train_mask]
        weights = train_ridge_translator(
            x_train, target_proto_train, cfg.ridge_lambda, device
        )
        translated = apply_translator(x_all, weights)
        centroid_scores_all = normalize_np(x_all) @ layer_prototypes[layer].T
        ridge_scores_all = translated @ prototypes.T
        raw_scores_all = normalize_np(x_all) @ prototypes.T
        mean_scores_all = normalize_np(x_all) @ mean_embeddings.T

        # Explicit final-subtoken recoverability diagnostic over candidate final IDs.
        token_scores_all = np.zeros((len(labels), len(unique_final_ids)), dtype=np.float32)
        for col, tok_id in enumerate(unique_final_ids):
            class_rows = np.where(final_token_ids == tok_id)[0]
            token_embedding_rows_for_id = final_token_embeddings[class_rows].mean(
                axis=0, keepdims=True
            )
            token_scores_all[:, col] = (normalize_np(x_all) @ normalize_np(token_embedding_rows_for_id).T)[:, 0]
        actual_final_cols = final_token_cols[labels]

        for split_name, mask in split_masks.items():
            split_labels = labels[mask]
            method_scores = {
                "implicit_token_lens": centroid_scores_all[mask],
                "ridge_to_final_implicit_lens": ridge_scores_all[mask],
                "raw_final_prototype": raw_scores_all[mask],
                "mean_subtoken_embedding": mean_scores_all[mask],
            }
            for method, scores in method_scores.items():
                metrics = metrics_from_scores(scores, split_labels)
                rows.append(
                    {
                        "method": method,
                        "split": split_name,
                        "layer": layer,
                        "top1": metrics["top1"],
                        "top5": metrics["top5"],
                        "mrr": metrics["mrr"],
                    }
                )
                detailed[split_name].setdefault(layer, {})[method] = {
                    "preds": metrics["preds"],
                    "ranks": metrics["ranks"],
                    "correct": metrics["preds"] == split_labels,
                }

            token_preds = np.argmax(token_scores_all[mask], axis=1)
            token_correct = token_preds == actual_final_cols[mask]
            rows.append(
                {
                    "method": "explicit_final_subtoken_embedding",
                    "split": split_name,
                    "layer": layer,
                    "top1": float(np.mean(token_correct)),
                    "top5": float("nan"),
                    "mrr": float("nan"),
                }
            )
            detailed[split_name].setdefault(layer, {})[
                "explicit_final_subtoken_embedding"
            ] = {
                "preds": token_preds,
                "ranks": np.full_like(token_preds, fill_value=-1),
                "correct": token_correct,
            }

    return rows, detailed


def evaluate_static_baselines(
    examples: list[dict[str, Any]], labels: np.ndarray, split: np.ndarray, n_classes: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    train_mask = split == "train"
    class_counts = Counter(labels[train_mask].tolist())
    majority_label = class_counts.most_common(1)[0][0]
    token_to_label_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for ex, y, is_train in zip(examples, labels, train_mask):
        if is_train:
            token_to_label_counts[int(ex["final_token_id"])][int(y)] += 1
    token_to_majority = {
        tok: counts.most_common(1)[0][0] for tok, counts in token_to_label_counts.items()
    }

    rows: list[dict[str, Any]] = []
    detailed: dict[str, dict[str, np.ndarray]] = {}
    for split_name in ("validation", "test"):
        mask = split == split_name
        y_true = labels[mask]
        split_examples = [ex for ex, use in zip(examples, mask) if use]
        majority_preds = np.full(len(y_true), fill_value=majority_label)
        subtoken_preds = np.array(
            [
                token_to_majority.get(int(ex["final_token_id"]), majority_label)
                for ex in split_examples
            ],
            dtype=np.int64,
        )
        for method, preds in (
            ("majority_word", majority_preds),
            ("final_subtoken_majority", subtoken_preds),
        ):
            rows.append(
                {
                    "method": method,
                    "split": split_name,
                    "layer": -1,
                    "top1": float(np.mean(preds == y_true)),
                    "top5": float("nan"),
                    "mrr": float("nan"),
                }
            )
            detailed.setdefault(split_name, {})[method] = {
                "preds": preds,
                "correct": preds == y_true,
            }
    return rows, detailed


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_randomization_p(
    a_correct: np.ndarray,
    b_correct: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> float:
    diffs = a_correct.astype(np.float32) - b_correct.astype(np.float32)
    observed = abs(float(diffs.mean()))
    if observed == 0.0:
        return 1.0
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_samples, len(diffs)))
    sampled = np.abs((signs * diffs[None, :]).mean(axis=1))
    return float((np.sum(sampled >= observed) + 1) / (n_samples + 1))


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict[str, Any]:
    a = a_correct.astype(bool)
    b = b_correct.astype(bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    p = 1.0
    if discordant > 0:
        p = float(stats.binomtest(min(a_only, b_only), discordant, p=0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": p}


def select_primary_comparison(
    rows: list[dict[str, Any]], detailed: dict[str, dict[int, dict[str, np.ndarray]]], static_detail: dict[str, dict[str, np.ndarray]], cfg: Config
) -> dict[str, Any]:
    val_rows = [row for row in rows if row["split"] == "validation"]
    itl_candidates = [row for row in val_rows if row["method"] == "implicit_token_lens"]
    best_itl = max(itl_candidates, key=lambda row: (row["top1"], row["mrr"]))
    baseline_methods = {"raw_final_prototype", "mean_subtoken_embedding", "majority_word", "final_subtoken_majority"}
    baseline_candidates = [row for row in val_rows if row["method"] in baseline_methods]
    best_baseline = max(baseline_candidates, key=lambda row: (row["top1"], row["mrr"] if not math.isnan(row["mrr"]) else -1.0))

    test_rows = [row for row in rows if row["split"] == "test"]
    def find_test_row(method: str, layer: int) -> dict[str, Any]:
        for row in test_rows:
            if row["method"] == method and int(row["layer"]) == int(layer):
                return row
        raise KeyError((method, layer))

    itl_test = find_test_row("implicit_token_lens", int(best_itl["layer"]))
    baseline_test = find_test_row(best_baseline["method"], int(best_baseline["layer"]))

    if best_baseline["method"] in static_detail["test"]:
        b_correct = static_detail["test"][best_baseline["method"]]["correct"]
    else:
        b_correct = detailed["test"][int(best_baseline["layer"])][best_baseline["method"]]["correct"]
    a_correct = detailed["test"][int(best_itl["layer"])]["implicit_token_lens"]["correct"]

    rng = np.random.default_rng(cfg.seed + 7)
    diff_values = a_correct.astype(np.float32) - b_correct.astype(np.float32)
    ci_low, ci_high = bootstrap_ci(diff_values, rng, cfg.bootstrap_samples)
    acc_ci = bootstrap_ci(a_correct.astype(np.float32), rng, cfg.bootstrap_samples)
    p_rand = paired_randomization_p(a_correct, b_correct, rng, cfg.randomization_samples)
    mcnemar = mcnemar_exact(a_correct, b_correct)

    return {
        "best_itl_validation": best_itl,
        "best_baseline_validation": best_baseline,
        "itl_test": itl_test,
        "baseline_test": baseline_test,
        "accuracy_difference_itl_minus_baseline": float(diff_values.mean()),
        "difference_bootstrap_ci95": [ci_low, ci_high],
        "itl_accuracy_bootstrap_ci95": list(acc_ci),
        "paired_randomization_p_value": p_rand,
        "mcnemar_exact": mcnemar,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=np.float64)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = n - rank + 1
        val = min(prev, p_values[idx] * n / original_rank)
        adjusted[idx] = val
        prev = val
    return [float(min(1.0, x)) for x in adjusted]


def layerwise_itl_vs_raw_tests(
    detailed: dict[str, dict[int, dict[str, np.ndarray]]], cfg: Config
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 11)
    tests = []
    layers = sorted(detailed["test"].keys())
    p_values = []
    for layer in layers:
        a = detailed["test"][layer]["implicit_token_lens"]["correct"]
        b = detailed["test"][layer]["raw_final_prototype"]["correct"]
        p = paired_randomization_p(a, b, rng, max(1000, cfg.randomization_samples // 3))
        p_values.append(p)
        tests.append(
            {
                "layer": layer,
                "diff_itl_minus_raw": float(np.mean(a.astype(np.float32) - b.astype(np.float32))),
                "p_value": p,
            }
        )
    adjusted = benjamini_hochberg(p_values)
    for row, adj in zip(tests, adjusted):
        row["bh_fdr_p_value"] = adj
    return tests


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_figures(rows: list[dict[str, Any]], figure_dir: Path, n_layers: int) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = [
        "implicit_token_lens",
        "ridge_to_final_implicit_lens",
        "raw_final_prototype",
        "mean_subtoken_embedding",
        "explicit_final_subtoken_embedding",
    ]
    labels = {
        "implicit_token_lens": "Implicit token lens",
        "ridge_to_final_implicit_lens": "Ridge-to-final ITL",
        "raw_final_prototype": "Raw final prototype",
        "mean_subtoken_embedding": "Mean subtoken embedding",
        "explicit_final_subtoken_embedding": "Explicit final-subtoken",
    }
    colors = {
        "implicit_token_lens": "#1b9e77",
        "ridge_to_final_implicit_lens": "#66a61e",
        "raw_final_prototype": "#7570b3",
        "mean_subtoken_embedding": "#d95f02",
        "explicit_final_subtoken_embedding": "#666666",
    }

    for metric, ylabel, filename in (
        ("top1", "Top-1 accuracy", "layer_top1_accuracy.png"),
        ("top5", "Top-5 accuracy", "layer_top5_accuracy.png"),
        ("mrr", "Mean reciprocal rank", "layer_mrr.png"),
    ):
        plt.figure(figsize=(9.0, 5.2))
        for method in methods:
            xs, ys = [], []
            for layer in range(n_layers):
                vals = [
                    row[metric]
                    for row in rows
                    if row["split"] == "test"
                    and row["method"] == method
                    and int(row["layer"]) == layer
                ]
                if vals and not math.isnan(float(vals[0])):
                    xs.append(layer)
                    ys.append(float(vals[0]))
            if xs:
                plt.plot(xs, ys, marker="o", linewidth=2, markersize=4, label=labels[method], color=colors[method])
        plt.xlabel("Layer (0 = embeddings)")
        plt.ylabel(ylabel)
        plt.title(f"Implicit-token decoding by layer ({metric})")
        plt.ylim(bottom=0.0)
        plt.grid(alpha=0.25)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=180)
        plt.close()


def save_error_examples(
    path: Path,
    examples: list[dict[str, Any]],
    final_meta: list[dict[str, Any]],
    labels: np.ndarray,
    split: np.ndarray,
    detailed: dict[str, dict[int, dict[str, np.ndarray]]],
    primary: dict[str, Any],
    max_examples: int = 40,
) -> None:
    layer = int(primary["best_itl_validation"]["layer"])
    test_indices = np.where(split == "test")[0]
    preds = detailed["test"][layer]["implicit_token_lens"]["preds"]
    ranks = detailed["test"][layer]["implicit_token_lens"]["ranks"]
    rows = []
    for local_i, global_i in enumerate(test_indices):
        if preds[local_i] == labels[global_i]:
            continue
        rows.append(
            {
                "example_id": int(examples[global_i]["example_id"]),
                "target": examples[global_i]["word"],
                "predicted": final_meta[int(preds[local_i])]["word"],
                "rank": int(ranks[local_i]),
                "context_tail": examples[global_i]["context"][-180:],
                "target_pieces": examples[global_i]["pieces"],
                "final_token": examples[global_i]["final_token"],
            }
        )
        if len(rows) >= max_examples:
            break
    path.write_text(json.dumps(rows, indent=2))


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    figure_dir = Path(cfg.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    env_info = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "numpy": np.__version__,
    }
    if torch.cuda.is_available():
        env_info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "mem_get_info_bytes": list(torch.cuda.mem_get_info(i)),
            }
            for i in range(torch.cuda.device_count())
        ]

    tokenizer, model, device = load_model_and_tokenizer(cfg)
    print(f"Using device: {device}", flush=True)
    occurrences = extract_all_occurrences(Path(cfg.dataset_path))
    selected_occurrences, word_meta = choose_candidate_words(occurrences, tokenizer, cfg)
    examples, final_meta = make_balanced_split(selected_occurrences, word_meta, cfg)
    if len(final_meta) < 30:
        raise RuntimeError(f"Too few candidate implicit words: {len(final_meta)}")

    labels = np.array([ex["label"] for ex in examples], dtype=np.int64)
    split = np.array([ex["split"] for ex in examples])
    n_classes = len(final_meta)

    (output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (output_dir / "environment.json").write_text(json.dumps(env_info, indent=2))
    (output_dir / "candidate_vocab.json").write_text(json.dumps(final_meta, indent=2))
    with (output_dir / "examples.jsonl").open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    safe_model = cfg.model_name.replace("/", "_").replace(".", "p")
    cache_path = output_dir / (
        f"hidden_states_{safe_model}_words{n_classes}_ex{len(examples)}"
        f"_seed{cfg.seed}_len{cfg.max_length}.npz"
    )
    if cache_path.exists():
        print(f"Loading cached hidden states: {cache_path}", flush=True)
        cache = np.load(cache_path)
        hidden_states = cache["hidden_states"]
    else:
        hidden_states = collect_hidden_states(examples, tokenizer, model, device, cfg)
        np.savez(
            cache_path,
            hidden_states=hidden_states,
            labels=labels,
            split=split,
        )

    mean_embeddings, final_token_embeddings = candidate_embedding_tables(final_meta, model)
    prototypes = compute_prototypes(hidden_states, labels, split, n_classes)
    layer_prototypes = compute_layer_prototypes(hidden_states, labels, split, n_classes)
    np.savez(
        output_dir / "implicit_vocab_tables.npz",
        prototypes=prototypes,
        layer_prototypes=layer_prototypes,
        mean_subtoken_embeddings=mean_embeddings,
        final_token_embeddings=final_token_embeddings,
    )

    layer_rows, detailed = evaluate_layer_scores(
        hidden_states,
        labels,
        split,
        prototypes,
        layer_prototypes,
        mean_embeddings,
        final_token_embeddings,
        final_meta,
        cfg,
        device,
    )
    static_rows, static_detail = evaluate_static_baselines(examples, labels, split, n_classes)
    rows = layer_rows + static_rows
    save_csv(output_dir / "metrics_layerwise.csv", rows)
    make_figures(rows, figure_dir, hidden_states.shape[1])

    primary = select_primary_comparison(rows, detailed, static_detail, cfg)
    layer_tests = layerwise_itl_vs_raw_tests(detailed, cfg)
    summary = {
        "config": asdict(cfg),
        "n_examples": len(examples),
        "n_candidate_words": n_classes,
        "split_counts": {name: int(np.sum(split == name)) for name in ("train", "validation", "test")},
        "n_layers_including_embedding": int(hidden_states.shape[1]),
        "hidden_size": int(hidden_states.shape[2]),
        "primary_comparison": primary,
        "layerwise_itl_vs_raw_tests": layer_tests,
        "candidate_vocab_stats": {
            "mean_piece_count": float(np.mean([len(meta["piece_ids"]) for meta in final_meta])),
            "min_piece_count": int(min(len(meta["piece_ids"]) for meta in final_meta)),
            "max_piece_count": int(max(len(meta["piece_ids"]) for meta in final_meta)),
            "unique_final_subtokens": int(len(set(meta["final_token_id"] for meta in final_meta))),
            "largest_final_subtoken_group": int(
                max(Counter(meta["final_token_id"] for meta in final_meta).values())
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    save_error_examples(
        output_dir / "error_examples.json",
        examples,
        final_meta,
        labels,
        split,
        detailed,
        primary,
    )

    validation = {
        "hidden_states_shape": list(hidden_states.shape),
        "all_hidden_finite": bool(np.isfinite(hidden_states).all()),
        "prototype_shape": list(prototypes.shape),
        "all_prototypes_finite": bool(np.isfinite(prototypes).all()),
        "all_splits_present": all(np.sum(split == name) > 0 for name in ("train", "validation", "test")),
        "n_candidate_words": n_classes,
        "outputs": [
            str(output_dir / "metrics_layerwise.csv"),
            str(output_dir / "summary.json"),
            str(figure_dir / "layer_top1_accuracy.png"),
            str(figure_dir / "layer_mrr.png"),
        ],
    }
    (output_dir / "validation_checks.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(summary["primary_comparison"], indent=2), flush=True)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-words", type=int, default=Config.max_words)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--model-name", default=Config.model_name)
    parser.add_argument("--device", default=Config.device)
    parser.add_argument("--ridge-lambda", type=float, default=Config.ridge_lambda)
    args = parser.parse_args()
    cfg = Config()
    cfg.max_words = args.max_words
    cfg.batch_size = args.batch_size
    cfg.model_name = args.model_name
    cfg.device = args.device
    cfg.ridge_lambda = args.ridge_lambda
    return cfg


if __name__ == "__main__":
    run(parse_args())
