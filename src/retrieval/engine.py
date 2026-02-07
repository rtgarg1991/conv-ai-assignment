from typing import List, Dict, Tuple
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.retrieval.vector_index import VectorIndex
from src.retrieval.sparse_index import SparseIndex
from src.retrieval.rrf import RRFGrouper


class HybridRetriever:
    def __init__(
        self,
        k_retrieval: int = 100,
        k_rrf: int = 60,
        k_final: int = Config.TOP_N_RETRIEVAL,
    ):
        self.k_retrieval = k_retrieval
        self.k_final = k_final

        self.vector_index = VectorIndex()
        self.sparse_index = SparseIndex()
        self.rrf_grouper = RRFGrouper(
            k_const=k_rrf,
            weight_dense=Config.RRF_WEIGHT_DENSE,
            weight_sparse=Config.RRF_WEIGHT_SPARSE,
        )

        self.output_log = []

    def initialize(self):
        """Loads or builds all indices."""
        print("Initializing Hybrid Retriever...")

        # Load Vector Index
        if Config.VECTOR_DB_PATH.exists():
            self.vector_index.load_index()
        else:
            print("Vector Index not found, building...")
            self.vector_index.build_index()

        # Load Sparse Index
        if (Config.DATA_DIR / "bm25_index.pkl").exists():
            self.sparse_index.load_index()
        else:
            print("Sparse Index not found, building...")
            self.sparse_index.build_index()

        print("Hybrid Retriever Initialized.")

    def retrieve(self, query: str) -> List[Tuple[Dict, float]]:
        """
        Performs hybrid retrieval for the query.
        """
        # Dense Search
        dense_results = self.vector_index.search(query, k=self.k_retrieval)

        # Sparse Search
        sparse_results = self.sparse_index.search(query, k=self.k_retrieval)

        # RRF
        final_results = self.rrf_grouper.fuse(
            dense_results, sparse_results, top_n_out=self.k_final
        )

        return final_results

    def retrieve_with_details(self, query: str) -> Dict:
        """
        Performs hybrid retrieval and returns detailed scores for each method.

        Returns:
            Dict with 'final_results', 'dense_results', 'sparse_results',
            and score breakdowns for UI display.
        """
        import time

        # Dense Search with timing
        start = time.time()
        dense_results = self.vector_index.search(query, k=self.k_retrieval)
        dense_time = time.time() - start

        # Sparse Search with timing
        start = time.time()
        sparse_results = self.sparse_index.search(query, k=self.k_retrieval)
        sparse_time = time.time() - start

        # RRF with timing
        start = time.time()
        final_results = self.rrf_grouper.fuse(
            dense_results, sparse_results, top_n_out=self.k_final
        )
        rrf_time = time.time() - start

        # Build score lookup tables
        dense_scores = {
            chunk["chunk_id"]: score for chunk, score in dense_results
        }
        sparse_scores = {
            chunk["chunk_id"]: score for chunk, score in sparse_results
        }

        # Enrich final results with individual scores
        enriched_results = []
        for chunk, rrf_score in final_results:
            chunk_id = chunk["chunk_id"]
            enriched_results.append(
                {
                    **chunk,
                    "rrf_score": round(rrf_score, 4),
                    "dense_score": round(dense_scores.get(chunk_id, 0), 4),
                    "sparse_score": round(sparse_scores.get(chunk_id, 0), 4),
                }
            )

        return {
            "final_results": enriched_results,
            "timing": {
                "dense_ms": round(dense_time * 1000, 2),
                "sparse_ms": round(sparse_time * 1000, 2),
                "rrf_ms": round(rrf_time * 1000, 2),
                "total_ms": round(
                    (dense_time + sparse_time + rrf_time) * 1000, 2
                ),
            },
        }


if __name__ == "__main__":
    retriever = HybridRetriever()
    retriever.initialize()

    query = "What is the central concept of Stoicism?"
    results = retriever.retrieve(query)

    print(f"\nQuery: {query}")
    print(f"Top {len(results)} Results:")
    for chunk, score in results:
        print(f"[{score:.4f}] {chunk['title']}: {chunk['content'][:100]}...")
