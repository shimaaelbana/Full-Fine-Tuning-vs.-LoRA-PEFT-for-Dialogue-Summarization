"""Full fine-tuning and LoRA (PEFT) fine-tuning routines for FLAN-T5.

Both routines return the trained model plus timing/parameter-count metadata,
so results are directly comparable in a single results table.
"""
import time
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .params import count_trainable_parameters


def _train(
    model, tokenizer, train_dataset, eval_dataset, output_dir: str,
    learning_rate: float, num_train_epochs: int, per_device_train_batch_size: int,
    gradient_accumulation_steps: int, gradient_checkpointing: bool,
) -> float:
    """Run a Trainer.train() call and return elapsed wall-clock seconds.

    Uses DataCollatorForSeq2Seq for dynamic per-batch padding (instead of a
    fixed 512-token pad for every example) and masks padded label positions
    with -100 so the loss doesn't train the model to predict padding. Mixed
    precision (fp16) and gradient accumulation keep this within a free-tier
    GPU's memory budget.
    """
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100,
    )
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        fp16=torch.cuda.is_available(),
        weight_decay=0.01,
        logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, data_collator=data_collator,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
    )
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    return elapsed


def full_finetune(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str = "./full-finetune-checkpoint",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
):
    """Fine-tune every parameter of the base model. Returns (model, metadata)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elapsed = _train(
        model, tokenizer, train_dataset, eval_dataset, output_dir,
        learning_rate, num_train_epochs, per_device_train_batch_size,
        gradient_accumulation_steps, gradient_checkpointing=True,
    )
    metadata = {"train_time_sec": elapsed, **count_trainable_parameters(model)}
    return model, metadata


def lora_finetune(
    model_name: str,
    train_dataset,
    eval_dataset,
    output_dir: str = "./lora-finetune-checkpoint",
    learning_rate: float = 1e-3,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
    lora_r: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
):
    """Fine-tune a LoRA adapter on top of a frozen base model.

    Returns (peft_model, metadata) where metadata includes the trainable
    parameter percentage -- this is the headline PEFT efficiency number.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["q", "v"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    peft_model = get_peft_model(base_model, lora_config)
    elapsed = _train(
        peft_model, tokenizer, train_dataset, eval_dataset, output_dir,
        learning_rate, num_train_epochs, per_device_train_batch_size,
        gradient_accumulation_steps, gradient_checkpointing=False,
    )
    metadata = {"train_time_sec": elapsed, "lora_r": lora_r, **count_trainable_parameters(peft_model)}
    return peft_model, metadata
