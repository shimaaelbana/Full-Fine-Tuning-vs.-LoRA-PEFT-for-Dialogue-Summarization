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

Trained on a 1000-example subset of DialogSum, 1 epoch, evaluated on 50
held-out test dialogues (`google/flan-t5-base`, single T4 GPU):

| Model | Trainable % | Train time (s) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|
| original (zero-shot) | 0% | 0 | 0.215 | 0.053 | 0.190 |
| full fine-tune | 100% | 735.0 | 0.339 | 0.086 | 0.269 |
| LoRA fine-tune (r=32) | 1.41% | 81.9 | 0.372 | 0.108 | 0.301 |

### LoRA rank sweep

| r | Trainable params | Trainable % | Train time (s) | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| 4  | 442,368   | 0.178% | 82.6 | 0.369 | 0.124 | 0.298 |
| 16 | 1,769,472 | 0.710% | 82.6 | 0.350 | 0.118 | 0.295 |
| 32 | 3,538,944 | 1.409% | 82.7 | 0.372 | 0.108 | 0.301 |

### Findings

- **LoRA didn't just approach full fine-tuning here — it beat it.** At
  r=32, LoRA reaches ROUGE-L 0.301 vs. full fine-tuning's 0.269, while
  training on **1.41% of the parameters** in **9x less wall-clock time**
  (81.9s vs. 735.0s for the same 1000 examples, 1 epoch). This is a smaller,
  short training run (1 epoch, 1000 examples), so it's plausible the full
  fine-tune is under-trained relative to its much larger effective capacity
  and would close the gap with more epochs or a larger learning-rate
  schedule tuned specifically for it — the comparison here holds
  hyperparameter budget roughly constant across methods rather than tuning
  each separately.
- **Both fine-tuning approaches clear zero-shot by a wide margin.**
  ROUGE-L roughly +0.08 (full FT) to +0.11 (LoRA) over the untouched base
  model — fine-tuning on in-domain data matters much more than which
  fine-tuning method is used, at this scale.
- **LoRA rank barely matters in this setup.** ROUGE-L is nearly flat from
  r=4 (0.298) to r=32 (0.301), and r=16 is actually the low point (0.295).
  With only ~430K–3.5M trainable parameters either way (well under 1.5% of
  the model), all three ranks likely have enough capacity to fit a
  1000-example training set — the bottleneck is the small data/epoch budget,
  not adapter capacity. r=4 is the practical pick here: same quality as
  r=32 at 8x fewer trainable parameters.
- **Training time is essentially identical across LoRA ranks** (~82.6s
  regardless of r=4 vs r=32), because at this scale the fixed cost of a
  forward/backward pass through the frozen 250M-parameter base dominates —
  the adapter's own parameter count is too small to move the needle on
  wall-clock time.

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
