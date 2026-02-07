"""
Error Analysis Module for Hybrid RAG System.

Analyzes evaluation results to categorize and understand failures:
- Retrieval failures: Correct source not in top-K
- Context failures: Right source but wrong chunk selected
- Generation failures: Good context but poor answer quality
- Unanswerable: Question outside corpus scope

Provides insights for system improvement.
"""

import json
import sys
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.generation.rag import RAGService
from src.evaluation.metrics import MetricsEvaluator


class ErrorAnalyzer:
    """
    Analyzes RAG system failures and categorizes errors.

    Error Categories:
    1. RETRIEVAL_MISS: Ground truth URL not in top-K retrieved
    2. RANK_DEGRADED: Ground truth found but not in top position
    3. LOW_ANSWER_QUALITY: Retrieved correctly but poor answer generation
    4. SUCCESS: Correct retrieval and good answer
    """

    def __init__(self):
        self.rag_service = RAGService()
        self.metrics = MetricsEvaluator()
        self.qa_path = Config.DATA_DIR / "qa_dataset.json"
        self.results_path = Config.DATA_DIR / "error_analysis.json"

    def initialize(self):
        """Initialize RAG service."""
        print("Initializing RAG service for error analysis...")
        self.rag_service.initialize()
        print("RAG service ready.")

    def load_qa_dataset(self) -> List[Dict]:
        """Load Q&A dataset."""
        with open(self.qa_path, "r") as f:
            return json.load(f)

    def analyze_errors(self, sample_size: int = None) -> Dict:
        """
        Run error analysis on Q&A dataset.

        Args:
            sample_size: Number of questions to analyze (None = all)

        Returns:
            Comprehensive error analysis report.
        """
        self.initialize()
        dataset = self.load_qa_dataset()

        if sample_size:
            dataset = dataset[:sample_size]

        print(f"\nAnalyzing errors on {len(dataset)} questions...")

        # Collect detailed results
        detailed_results = []
        error_counts = defaultdict(int)
        errors_by_type = defaultdict(list)

        for i, item in enumerate(dataset):
            query = item["question"]
            ground_truth_url = item["url"]
            question_type = item.get("question_type", "unknown")

            # Run RAG
            result = self.rag_service.answer_question(query)
            retrieved_urls = [c["url"] for c in result["retrieved_chunks"]]

            # Categorize result
            category, details = self._categorize_result(
                ground_truth_url,
                retrieved_urls,
                item.get("answer", ""),
                result["answer"],
            )

            error_counts[category] += 1

            record = {
                "question_id": i,
                "question": query[:100],
                "question_type": question_type,
                "category": category,
                "ground_truth_url": ground_truth_url,
                "retrieved_urls": retrieved_urls[:3],
                "generated_answer": result["answer"][:200],
                "details": details,
            }
            detailed_results.append(record)

            # Track errors by question type
            if category != "SUCCESS":
                errors_by_type[question_type].append(record)

            print(f"Analyzed: {i + 1}/{len(dataset)} - {category}", end="\r")

        print("\n\nAnalysis complete.")

        # Generate summary statistics
        total = len(dataset)
        summary = {
            "total_questions": total,
            "categories": {
                cat: {
                    "count": count,
                    "percentage": round(count / total * 100, 2),
                }
                for cat, count in error_counts.items()
            },
            "success_rate": round(
                error_counts.get("SUCCESS", 0) / total * 100, 2
            ),
            "retrieval_accuracy": round(
                (total - error_counts.get("RETRIEVAL_MISS", 0)) / total * 100,
                2,
            ),
        }

        # Errors by question type
        type_analysis = {}
        for qtype, errors in errors_by_type.items():
            type_analysis[qtype] = {
                "error_count": len(errors),
                "sample_errors": [e["question"][:50] for e in errors[:3]],
            }

        # Generate recommendations
        recommendations = self._generate_recommendations(error_counts, total)

        # Full report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "errors_by_question_type": type_analysis,
            "recommendations": recommendations,
            "detailed_results": detailed_results,
        }

        # Save
        with open(self.results_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Error analysis saved to {self.results_path}")

        # Print summary
        self._print_summary(summary, type_analysis, recommendations)

        return report

    def _categorize_result(
        self,
        ground_truth_url: str,
        retrieved_urls: List[str],
        reference_answer: str,
        generated_answer: str,
    ) -> Tuple[str, str]:
        """
        Categorize a single result.

        Returns:
            Tuple of (category, details)
        """
        # Check if ground truth URL is in retrieved results
        if ground_truth_url not in retrieved_urls:
            return "RETRIEVAL_MISS", "Ground truth URL not in top-K retrieved"

        # Check rank position
        rank = retrieved_urls.index(ground_truth_url) + 1

        if rank > 1:
            return (
                "RANK_DEGRADED",
                f"Ground truth found at rank {rank}, not #1",
            )

        # Check answer quality (simple heuristic)
        if len(generated_answer.strip()) < 10:
            return "LOW_ANSWER_QUALITY", "Generated answer too short"

        # Success
        return "SUCCESS", "Correct retrieval and reasonable answer"

    def _generate_recommendations(
        self, error_counts: Dict, total: int
    ) -> List[str]:
        """Generate improvement recommendations based on error patterns."""
        recommendations = []

        retrieval_miss_rate = error_counts.get("RETRIEVAL_MISS", 0) / total
        rank_degraded_rate = error_counts.get("RANK_DEGRADED", 0) / total

        if retrieval_miss_rate > 0.3:
            recommendations.append(
                "HIGH RETRIEVAL MISS RATE: Consider increasing top-K, "
                "improving embeddings, or adding query expansion"
            )

        if rank_degraded_rate > 0.2:
            recommendations.append(
                "RANK DEGRADATION: RRF weights may need tuning, "
                "or consider re-ranking model"
            )

        if error_counts.get("LOW_ANSWER_QUALITY", 0) > total * 0.1:
            recommendations.append(
                "ANSWER QUALITY: Consider using larger LLM or "
                "improving prompt engineering"
            )

        if not recommendations:
            recommendations.append(
                "System performing well. Consider edge case testing."
            )

        return recommendations

    def _print_summary(
        self, summary: Dict, type_analysis: Dict, recommendations: List[str]
    ):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("            ERROR ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"\nTotal Questions: {summary['total_questions']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Retrieval Accuracy: {summary['retrieval_accuracy']}%")

        print("\nError Breakdown:")
        print("-" * 40)
        for cat, data in summary["categories"].items():
            print(f"  {cat:<20} {data['count']:>5} ({data['percentage']}%)")

        if type_analysis:
            print("\nErrors by Question Type:")
            print("-" * 40)
            for qtype, data in type_analysis.items():
                print(f"  {qtype}: {data['error_count']} errors")

        print("\nRecommendations:")
        print("-" * 40)
        for rec in recommendations:
            print(f"  • {rec}")

        print("=" * 60)


if __name__ == "__main__":
    analyzer = ErrorAnalyzer()
    # Run on subset for quick analysis
    analyzer.analyze_errors(sample_size=20)
