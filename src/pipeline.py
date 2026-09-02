"""Orchestrates the full comparison: zero-shot baseline vs full fine-tuning
vs LoRA fine-tuning, plus an optional LoRA rank sweep.
"""
from typing import Dict, List

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .data import load_dialogsum, subsample, tokenize_dataset
from .evaluate import generate_summaries, score_summaries
from .params import count_trainable_parameters
from .train import full_finetune, lora_finetune


def run_comparison(
    model_name: str = "google/flan-t5-base",
    train_size: int = 1000,
    eval_size: int = 100,
    test_size: int = 50,
    num_train_epochs: int = 1,
    lora_r: int = 32,
) -> pd.DataFrame:
    """Train a full fine-tune and a LoRA fine-tune on the same data subset,
    evaluate both plus the untouched base model, and return one comparison
    table with ROUGE scores, trainable-parameter percentage, and training time.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = load_dialogsum()
    subset = subsample(raw, train_size, eval_size, test_size)

    tokenized_train = tokenize_dataset(subset["train"], tokenizer)
    tokenized_eval = tokenize_dataset(subset["validation"], tokenizer)

    test_dialogues = subset["test"]["dialogue"]
    test_references = subset["test"]["summary"]

    rows = []

    # Zero-shot baseline: untouched pretrained model, no fine-tuning at all.
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    base_predictions = generate_summaries(base_model, tokenizer, test_dialogues)
    rows.append({
        "model": "original (zero-shot)",
        "train_time_sec": 0.0,
        "trainable_pct": 0.0,
        **score_summaries(base_predictions, test_references),
    })

    # Full fine-tuning: every parameter is updated.
    full_model, full_meta = full_finetune(
        model_name, tokenized_train, tokenized_eval, num_train_epochs=num_train_epochs,
    )
    full_predictions = generate_summaries(full_model, tokenizer, test_dialogues)
    rows.append({
        "model": "full fine-tune",
        "train_time_sec": full_meta["train_time_sec"],
        "trainable_pct": full_meta["trainable_pct"],
        **score_summaries(full_predictions, test_references),
    })

    # LoRA fine-tuning: only adapter weights are updated.
    lora_model, lora_meta = lora_finetune(
        model_name, tokenized_train, tokenized_eval,
        num_train_epochs=num_train_epochs, lora_r=lora_r,
    )
    lora_predictions = generate_summaries(lora_model, tokenizer, test_dialogues)
    rows.append({
        "model": f"LoRA fine-tune (r={lora_r})",
        "train_time_sec": lora_meta["train_time_sec"],
        "trainable_pct": lora_meta["trainable_pct"],
        **score_summaries(lora_predictions, test_references),
    })

    return pd.DataFrame(rows)


def run_lora_rank_sweep(
    model_name: str = "google/flan-t5-base",
    ranks: List[int] = (4, 16, 32),
    train_size: int = 1000,
    eval_size: int = 100,
    test_size: int = 50,
    num_train_epochs: int = 1,
) -> pd.DataFrame:
    """Original extension beyond the source lab: how does LoRA rank trade off
    parameter efficiency against summarization quality? Trains one LoRA
    adapter per rank on the same data and returns one row per rank.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    raw = load_dialogsum()
    subset = subsample(raw, train_size, eval_size, test_size)

    tokenized_train = tokenize_dataset(subset["train"], tokenizer)
    tokenized_eval = tokenize_dataset(subset["validation"], tokenizer)
    test_dialogues = subset["test"]["dialogue"]
    test_references = subset["test"]["summary"]

    rows = []
    for r in ranks:
        model, meta = lora_finetune(
            model_name, tokenized_train, tokenized_eval,
            num_train_epochs=num_train_epochs, lora_r=r,
        )
        predictions = generate_summaries(model, tokenizer, test_dialogues)
        rows.append({
            "lora_r": r,
            "trainable_params": meta["trainable_params"],
            "trainable_pct": meta["trainable_pct"],
            "train_time_sec": meta["train_time_sec"],
            **score_summaries(predictions, test_references),
        })

    return pd.DataFrame(rows)
