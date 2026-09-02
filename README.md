# Full Fine-Tuning vs. LoRA (PEFT) for Dialogue Summarization

Fine-tunes **FLAN-T5-base (250M)** on [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)
two ways — updating every parameter ("full fine-tuning") vs. training a
frozen-base **LoRA adapter** — and compares them on summarization quality
(ROUGE), training time, and trainable-parameter percentage. Every model here
is trained from scratch in this repo; nothing is a downloaded pre-trained
checkpoint.

## Motivation

Full fine-tuning and LoRA are usually compared with the framing "LoRA is
almost as good, for a fraction of the parameters" — but "almost as good" is
rarely quantified against training time and a proper zero-shot baseline in
the same table. This project puts all three (zero-shot, full fine-tune,
LoRA) on the same test set with the same metrics, and adds a rank sweep to
show *how* the LoRA efficiency/quality tradeoff actually moves as `r` changes.

## Method

- **Dataset:** DialogSum, subsampled to a small train/validation/test split
  (sizes configurable; default 1000/100/50) to keep training time reasonable
  on a single free-tier GPU.
- **Model:** `google/flan-t5-base`.
- **Full fine-tuning:** all ~250M parameters updated via Hugging Face
  `Trainer`.
- **LoRA fine-tuning:** `peft` `LoraConfig` targeting the `q`/`v` attention
  projections, base model frozen (`r=32` by default).
- **LoRA rank sweep:** LoRA adapters trained at `r ∈ {4, 16, 32}` to show the
  parameter-count vs. ROUGE tradeoff as rank increases — not covered by the
  source lab this project is derived from.
- **Evaluation:** ROUGE-1/2/L F-measure (`rouge-score`) on a held-out test
  subset, plus trainable-parameter percentage and wall-clock training time
  for each approach.

## Results

_Run `kaggle/finetune_comparison.ipynb` to regenerate
`results/comparison.csv` and `results/lora_rank_sweep.csv`, then fill in:_

| Model | Trainable % | Train time (s) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|
| original (zero-shot) | 0% | 0 | | | |
| full fine-tune | 100% | | | | |
| LoRA fine-tune (r=32) | | | | | |

### LoRA rank sweep

| r | Trainable params | Trainable % | ROUGE-L |
|---|---|---|---|
| 4 | | | |
| 16 | | | |
| 32 | | | |

## Repository structure

```
src/
  data.py       # DialogSum loading, prompt building, tokenization, subsampling
  params.py     # trainable-parameter counting
  train.py      # full_finetune() and lora_finetune() routines
  evaluate.py   # generation + ROUGE scoring
  pipeline.py   # run_comparison() and run_lora_rank_sweep()
kaggle/
  finetune_comparison.ipynb   # self-contained Kaggle notebook (needs a GPU)
results/
  comparison.csv
  lora_rank_sweep.csv
requirements.txt
```

## Running it

**On Kaggle (recommended — needs a GPU):** open
`kaggle/finetune_comparison.ipynb` as a new Kaggle notebook with a GPU
accelerator enabled and Internet turned on, then run all cells. It clones
this repo and reuses the same `src/` code.

**Locally with a GPU:**

```bash
pip install -r requirements.txt
python -c "
from src.pipeline import run_comparison
df = run_comparison()
df.to_csv('results/comparison.csv', index=False)
print(df)
"
```

## Attribution

Dataset: [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)
(Chen et al.). Model: [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)
(Google, via Hugging Face `transformers`). PEFT/LoRA:
[Hugging Face `peft`](https://github.com/huggingface/peft).

## License

MIT — see [LICENSE](LICENSE).
