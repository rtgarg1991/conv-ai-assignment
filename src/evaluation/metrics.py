from typing import List, Dict, Union
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import torch


class MetricsEvaluator:
    def __init__(self):
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=True
        )
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def calculate_mrr(
        self,
        ground_truth_ids: List[str],
        retrieved_results_lists: List[List[Dict]],
    ) -> float:
        """
        Calculates Mean Reciprocal Rank (MRR).
        ground_truth_ids: List of chunk_ids or URLs that are correct for each query.
        retrieved_results_lists: List of lists of retrieved items (top-k) for each query.
        """
        reciprocal_ranks = []

        for gt_id, results in zip(ground_truth_ids, retrieved_results_lists):
            rank = 0

            # URL-level MRR: collapse to unique URL ranks if gt_id is a URL
            if isinstance(gt_id, str) and gt_id.startswith("http"):
                seen_urls = set()
                url_rank = 0
                for item in results:
                    url = item.get("url")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    url_rank += 1
                    if url == gt_id:
                        rank = url_rank
                        break
            else:
                # Chunk-level fallback
                for i, item in enumerate(results):
                    if item.get("chunk_id") == gt_id:
                        rank = i + 1
                        break

            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    def calculate_rouge(
        self, references: List[str], hypotheses: List[str]
    ) -> float:
        """Calculates average ROUGE-L F1 score."""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = self.rouge_scorer.score(ref, hyp)
            scores.append(score["rougeL"].fmeasure)
        return np.mean(scores) if scores else 0.0

    def calculate_rouge_list(
        self, references: List[str], hypotheses: List[str]
    ) -> List[float]:
        """Calculates per-example ROUGE-L F1 scores."""
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = self.rouge_scorer.score(ref, hyp)
            scores.append(score["rougeL"].fmeasure)
        return scores

    def calculate_bertscore(
        self, references: List[str], hypotheses: List[str]
    ) -> float:
        """Calculates average BERTScore F1."""
        # Check lengths
        if not references or not hypotheses:
            return 0.0

        P, R, F1 = score(
            hypotheses,
            references,
            lang="en",
            verbose=False,
            device=self.device,
            model_type="distilbert-base-uncased",
        )
        return F1.mean().item()

    def calculate_bertscore_list(
        self, references: List[str], hypotheses: List[str]
    ) -> List[float]:
        """Calculates per-example BERTScore F1."""
        if not references or not hypotheses:
            return []

        P, R, F1 = score(
            hypotheses,
            references,
            lang="en",
            verbose=False,
            device=self.device,
            model_type="distilbert-base-uncased",
        )
        return [v.item() for v in F1]


if __name__ == "__main__":
    # Unit Test
    evaluator = MetricsEvaluator()

    # MRR Test
    gt = ["A", "B"]
    retrieved = [
        [{"chunk_id": "C"}, {"chunk_id": "A"}],  # Rank 2 -> 0.5
        [{"chunk_id": "B"}, {"chunk_id": "D"}],  # Rank 1 -> 1.0
    ]
    mrr = evaluator.calculate_mrr(gt, retrieved)
    print(f"MRR Test (Exp 0.75): {mrr}")
    assert mrr == 0.75

    # ROUGE Test
    ref = ["The capital of France is Paris."]
    hyp = ["Paris is the capital of France."]
    rouge = evaluator.calculate_rouge(ref, hyp)
    print(f"ROUGE Test: {rouge}")

    # BERTScore Test
    bert = evaluator.calculate_bertscore(ref, hyp)
    print(f"BERTScore Test: {bert}")
