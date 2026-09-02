"""Data loading and tokenization for supervised dialogue-summarization fine-tuning."""
from datasets import load_dataset

DATASET_NAME = "knkarthick/dialogsum"

START_PROMPT = "Summarize the following conversation.\n\n"
END_PROMPT = "\n\nSummary: "


def load_dialogsum():
    """Load the DialogSum dataset (train/validation/test splits) from Hugging Face."""
    return load_dataset(DATASET_NAME)


def build_prompt(dialogue: str) -> str:
    """Build the instruction prompt used for both training and inference."""
    return f"{START_PROMPT}{dialogue}{END_PROMPT}"


def tokenize_dataset(dataset, tokenizer, max_length: int = 512):
    """Tokenize dialogue -> input_ids and summary -> labels for every split.

    Truncates (but does not pad) to `max_length`; padding is left to a
    dynamic data collator at training time so batches only pad to the
    longest sequence in that batch, not to a fixed 512 for every example.
    Returns a DatasetDict with the original text columns removed.
    """

    def _tokenize(example):
        prompts = [build_prompt(d) for d in example["dialogue"]]
        model_inputs = tokenizer(prompts, truncation=True, max_length=max_length)
        labels = tokenizer(example["summary"], truncation=True, max_length=max_length)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        _tokenize, batched=True,
        remove_columns=["id", "topic", "dialogue", "summary"],
    )


def subsample(dataset, train_size: int, eval_size: int, test_size: int, seed: int = 42):
    """Return a small, reproducible subset of each split for fast fine-tuning runs."""
    return {
        "train": dataset["train"].shuffle(seed=seed).select(range(train_size)),
        "validation": dataset["validation"].shuffle(seed=seed).select(range(eval_size)),
        "test": dataset["test"].shuffle(seed=seed).select(range(test_size)),
    }
