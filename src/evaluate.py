"""Generate summaries from a fine-tuned model and score them with ROUGE."""
from typing import Dict, List

from rouge_score import rouge_scorer

from .data import build_prompt


def generate_summaries(model, tokenizer, dialogues: List[str], max_new_tokens: int = 200) -> List[str]:
    """Run greedy generation over a list of raw dialogue strings."""
    predictions = []
    for dialogue in dialogues:
        prompt = build_prompt(dialogue)
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids
        output_ids = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        predictions.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return predictions


def score_summaries(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute average ROUGE-1 / ROUGE-2 / ROUGE-L F-measure across a batch."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    n = max(len(predictions), 1)
    return {key: value / n for key, value in totals.items()}
