import json
import tqdm
import sys
import os
import time
import pandas as pd
from typing import List, Dict
from datetime import datetime

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
        failed_count = 0
        total_latency = 0.0

        # Ground Truth / Predictions Lists for Aggregate Metrics
        ground_truth_urls = []
        retrieved_results = []
        ref_answers = []
        gen_answers = []

        print("Running Inference...")
        for item in tqdm.tqdm(dataset, desc="Evaluating"):
            query = item["question"]

            try:
                # Run RAG with timing
                start_time = time.time()
                rag_output = self.rag_service.answer_question(query)
                latency = time.time() - start_time
                total_latency += latency

                # Store for Metrics
                ground_truth_urls.append(item["url"])
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
                        "question_type": item.get("question_type", ""),
                        "ground_truth_url": item["url"],
                        "retrieved_urls": [
                            c["url"] for c in rag_output["retrieved_chunks"]
                        ],
                        "generated_answer": rag_output["answer"],
                        "reference_answer": item.get("answer", ""),
                        "latency_seconds": round(latency, 3),
                    }
                )
            except Exception as e:
                print(f"\nError processing question: {query[:50]}... - {e}")
                failed_count += 1
                continue

        print(f"\nProcessed {len(results)}/{len(dataset)} questions ({failed_count} failed)")
        
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
        
        avg_latency = total_latency / len(results) if results else 0.0

        print("\n" + "=" * 40)
        print("        EVALUATION RESULTS")
        print("=" * 40)
        print(f"  MRR (Retrieval):    {mrr_score:.4f}")
        print(f"  ROUGE-L (Gen):      {rouge_score:.4f}")
        print(f"  BERTScore (Gen):    {bert_score:.4f}")
        print(f"  Avg Latency:        {avg_latency:.3f}s")
        print(f"  Questions Eval'd:   {len(results)}")
        print("=" * 40)

        # Save detailed results to CSV
        df = pd.DataFrame(results)
        csv_path = Config.DATA_DIR / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to {csv_path}")
        
        # Save summary metrics to JSON
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(results),
            "num_failed": failed_count,
            "metrics": {
                "mrr": round(mrr_score, 4),
                "rouge_l": round(rouge_score, 4),
                "bert_score": round(bert_score, 4),
            },
            "latency": {
                "avg_seconds": round(avg_latency, 3),
                "total_seconds": round(total_latency, 3),
            },
            "config": {
                "embedding_model": Config.EMBEDDING_MODEL_NAME,
                "generation_model": Config.GENERATION_MODEL,
                "top_n_retrieval": Config.TOP_N_RETRIEVAL,
                "rrf_k": Config.RRF_K,
            }
        }
        json_path = Config.DATA_DIR / "evaluation_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary metrics saved to {json_path}")
        
        return summary


if __name__ == "__main__":
    runner = EvaluationRunner()
    runner.run_evaluation()
