import json
import tqdm
import sys
import os
import pandas as pd
from typing import List, Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.generation.rag import RAGService
from src.evaluation.metrics import MetricsEvaluator


class EvaluationRunner:
    def __init__(self):
        self.rag_service = RAGService()
        self.metrics_evaluator = MetricsEvaluator()
        self.qa_dataset_path = Config.DATA_DIR / "qa_dataset.json"

    def load_dataset(self) -> List[Dict]:
        with open(self.qa_dataset_path, "r") as f:
            return json.load(f)

    def run_evaluation(self):
        print("Initializing Evaluation Pipeline...")
        self.rag_service.initialize()

        dataset = self.load_dataset()
        print(f"Loaded {len(dataset)} Q&A pairs.")

        results = []

        # Ground Truth / Predictions Lists for Aggregate Metrics
        ground_truth_urls = []
        retrieved_results = []
        ref_answers = []
        gen_answers = []

        print("Running Inference...")
        for item in tqdm.tqdm(dataset):
            query = item["question"]

            # Run RAG
            rag_output = self.rag_service.answer_question(query)

            # Store for Metrics
            ground_truth_urls.append(
                item["url"]
            )  # Using URL as unique identifier
            retrieved_chunks_for_mrr = rag_output["retrieved_chunks"]
            retrieved_results.append(retrieved_chunks_for_mrr)

            ref_answers.append(
                item["answer"]
                if (item.get("answer") and len(item["answer"]) > 5)
                else item["ground_truth_context"]
            )
            gen_answers.append(rag_output["answer"])

            # Item Result
            results.append(
                {
                    "question": query,
                    "ground_truth_url": item["url"],
                    "retrieved_urls": [
                        c["url"] for c in rag_output["retrieved_chunks"]
                    ],
                    "generated_answer": rag_output["answer"],
                    "reference_answer": item.get("answer", ""),
                }
            )

        print("Calculating Metrics...")
        mrr_score = self.metrics_evaluator.calculate_mrr(
            ground_truth_urls, retrieved_results
        )
        rouge_score = self.metrics_evaluator.calculate_rouge(
            ref_answers, gen_answers
        )
        bert_score = self.metrics_evaluator.calculate_bertscore(
            ref_answers, gen_answers
        )

        print("\n" + "=" * 30)
        print("   EVALUATION RESULTS")
        print("=" * 30)
        print(f"MRR (Retrieval): {mrr_score:.4f}")
        print(f"ROUGE-L (Gen):   {rouge_score:.4f}")
        print(f"BERTScore (Gen): {bert_score:.4f}")
        print("=" * 30)

        # Save detailed results
        df = pd.DataFrame(results)
        output_path = Config.DATA_DIR / "evaluation_results.csv"
        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to {output_path}")


if __name__ == "__main__":
    runner = EvaluationRunner()
    runner.run_evaluation()
