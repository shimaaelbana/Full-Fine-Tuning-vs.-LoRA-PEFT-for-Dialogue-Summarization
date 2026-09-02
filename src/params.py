"""Utility for reporting how many parameters a model actually trains."""
from typing import Dict


def count_trainable_parameters(model) -> Dict[str, float]:
    """Return trainable / total parameter counts and the trainable percentage."""
    trainable, total = 0, 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total if total else 0.0,
    }
