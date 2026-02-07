"""
Ablation Studies Module for Hybrid RAG System.

Compares performance of different retrieval methods:
- Dense-only (FAISS vector search)
- Sparse-only (BM25)
- Hybrid (RRF with different k values)

This provides insights into the contribution of each component.
"""

import json
import sys
import os
from typing import List, Dict
from datetime import datetime
import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.retrieval.vector_index import VectorIndex
from src.retrieval.sparse_index import SparseIndex
from src.retrieval.rrf import RRFGrouper
from src.evaluation.metrics import MetricsEvaluator


class AblationStudy:
    """
    Ablation study comparing retrieval methods.

    Methods compared:
    1. Dense-only (FAISS cosine similarity)
    2. Sparse-only (BM25 keyword matching)
    3. Hybrid RRF with k=30 (gives more weight to top ranks)
    4. Hybrid RRF with k=60 (balanced)
    5. Hybrid RRF with k=100 (more uniform weighting)
    """

    def __init__(self):
        self.vector_index = VectorIndex()
        self.sparse_index = SparseIndex()
        self.metrics = MetricsEvaluator()
        self.qa_path = Config.DATA_DIR / "qa_dataset.json"
        self.results_path = Config.DATA_DIR / "ablation_results.json"

    def initialize(self):
        """Load all indices."""
        print("Initializing indices for ablation study...")

        if Config.VECTOR_DB_PATH.exists():
            self.vector_index.load_index()
        else:
            raise FileNotFoundError(
                "Vector index not found. Run data pipeline first."
            )

        if (Config.DATA_DIR / "bm25_index.pkl").exists():
            self.sparse_index.load_index()
        else:
            raise FileNotFoundError(
                "BM25 index not found. Run data pipeline first."
            )

        print("Indices loaded.")

    def load_qa_dataset(self) -> List[Dict]:
        """Load Q&A dataset for evaluation."""
        with open(self.qa_path, "r") as f:
            return json.load(f)

    def run_ablation(self, sample_size: int = None) -> Dict:
        """
        Run ablation study comparing all retrieval methods.

        Args:
            sample_size: Number of questions to use (None = all)

        Returns:
            Dictionary with results for each method.
        """
        self.initialize()
        dataset = self.load_qa_dataset()

        if sample_size:
            dataset = dataset[:sample_size]

        print(f"\nRunning ablation study on {len(dataset)} questions...")

        # Define methods to compare
        methods = [
            ("dense_only", self._retrieve_dense_only),
            ("sparse_only", self._retrieve_sparse_only),
            ("hybrid_k30", lambda q: self._retrieve_hybrid(q, k=30)),
            ("hybrid_k60", lambda q: self._retrieve_hybrid(q, k=60)),
            ("hybrid_k100", lambda q: self._retrieve_hybrid(q, k=100)),
        ]

        results = {}

        for method_name, retrieval_fn in methods:
            print(f"\n{'=' * 50}")
            print(f"Testing: {method_name}")
            print(f"{'=' * 50}")

            ground_truth_urls = []
            retrieved_results = []

            for item in tqdm.tqdm(dataset, desc=method_name):
                query = item["question"]
                chunks = retrieval_fn(query)

                ground_truth_urls.append(item["url"])
                retrieved_results.append(chunks)

            # Calculate MRR for this method
            mrr = self.metrics.calculate_mrr(
                ground_truth_urls, retrieved_results
            )

            results[method_name] = {
                "mrr": round(mrr, 4),
                "questions_evaluated": len(dataset),
            }
            print(f"MRR: {mrr:.4f}")

        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(dataset),
            "methods": results,
            "analysis": self._generate_analysis(results),
        }

        with open(self.results_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {self.results_path}")

        # Print summary
        self._print_summary(results)

        return output

    def _retrieve_dense_only(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve using only dense (vector) search."""
        results = self.vector_index.search(query, k=k)
        return [chunk for chunk, score in results]

    def _retrieve_sparse_only(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve using only sparse (BM25) search."""
        results = self.sparse_index.search(query, k=k)
        return [chunk for chunk, score in results]

    def _retrieve_hybrid(
        self, query: str, k: int = 60, top_n: int = 5
    ) -> List[Dict]:
        """Retrieve using hybrid RRF with specified k constant."""
        dense_results = self.vector_index.search(query, k=100)
        sparse_results = self.sparse_index.search(query, k=100)

        rrf = RRFGrouper(k_const=k)
        fused = rrf.fuse(dense_results, sparse_results, top_n_out=top_n)

        return [chunk for chunk, score in fused]

    def _generate_analysis(self, results: Dict) -> Dict:
        """Generate analysis comparing methods."""
        mrr_values = {method: data["mrr"] for method, data in results.items()}

        best_method = max(mrr_values, key=mrr_values.get)
        worst_method = min(mrr_values, key=mrr_values.get)

        # Calculate improvement of hybrid over single methods
        dense_mrr = mrr_values.get("dense_only", 0)
        sparse_mrr = mrr_values.get("sparse_only", 0)
        best_single = max(dense_mrr, sparse_mrr)
        best_hybrid = max(
            mrr_values.get("hybrid_k30", 0),
            mrr_values.get("hybrid_k60", 0),
            mrr_values.get("hybrid_k100", 0),
        )

        hybrid_improvement = (
            ((best_hybrid - best_single) / best_single * 100)
            if best_single > 0
            else 0
        )

        return {
            "best_method": best_method,
            "best_mrr": mrr_values[best_method],
            "worst_method": worst_method,
            "worst_mrr": mrr_values[worst_method],
            "dense_vs_sparse": "dense"
            if dense_mrr > sparse_mrr
            else "sparse",
            "hybrid_improvement_pct": round(hybrid_improvement, 2),
            "recommendation": f"Use {best_method} for optimal performance",
        }

    def _print_summary(self, results: Dict):
        """Print formatted summary table."""
        print("\n" + "=" * 50)
        print("         ABLATION STUDY SUMMARY")
        print("=" * 50)
        print(f"{'Method':<20} {'MRR':>10}")
        print("-" * 30)

        # Sort by MRR descending
        sorted_methods = sorted(
            results.items(), key=lambda x: x[1]["mrr"], reverse=True
        )

        for method, data in sorted_methods:
            print(f"{method:<20} {data['mrr']:>10.4f}")

        print("=" * 50)


if __name__ == "__main__":
    study = AblationStudy()
    # Run on all questions (or specify sample_size for quick test)
    study.run_ablation(sample_size=20)  # Use 20 for quick test
