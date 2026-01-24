"""
Reciprocal Rank Fusion (RRF) module.

Combines results from dense and sparse retrieval using RRF algorithm
with configurable k constant and weighting parameters.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config


class RRFGrouper:
    """
    Reciprocal Rank Fusion combiner for hybrid retrieval.

    Uses the formula: RRF_score(d) = sum(1 / (k + rank_i(d)))
    where k is a constant (default 60) and rank_i is the rank in retrieval system i.
    """

    def __init__(
        self,
        k_const: int = 60,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.0,
    ):
        """
        Initialize RRF grouper.

        Args:
            k_const: Constant for RRF formula (default 60).
            weight_dense: Weight for dense retrieval scores.
            weight_sparse: Weight for sparse retrieval scores.
        """
        self.k_const = k_const
        self.weight_dense = weight_dense
        self.weight_sparse = weight_sparse

    def fuse(
        self,
        dense_results: List[Tuple[Dict, float]],
        sparse_results: List[Tuple[Dict, float]],
        top_n_out: int = None,
        preserve_top_dense: int = 1,
    ) -> List[Tuple[Dict, float]]:
        """
        Combine results from Dense and Sparse retrieval using RRF.

        Args:
            dense_results: List of (chunk, score) from dense retrieval.
            sparse_results: List of (chunk, score) from sparse retrieval.
            top_n_out: Number of results to return.
            preserve_top_dense: Guarantee top N dense results are included.

        Returns:
            List of (chunk, rrf_score) sorted by score descending.
        """
        # Map chunks by ID and accumulate RRF scores
        chunk_map = {}
        rrf_scores = defaultdict(float)
        
        # Track top dense results to preserve
        top_dense_ids = set()
        for i, (chunk, score) in enumerate(dense_results[:preserve_top_dense]):
            top_dense_ids.add(chunk["chunk_id"])

        # Process Dense results (1-based ranking)
        for rank, (chunk, score) in enumerate(dense_results, 1):
            chunk_id = chunk["chunk_id"]
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] += self.weight_dense * (
                1 / (self.k_const + rank)
            )

        # Process Sparse results (1-based ranking)
        for rank, (chunk, score) in enumerate(sparse_results, 1):
            chunk_id = chunk["chunk_id"]
            chunk_map[chunk_id] = chunk
            rrf_scores[chunk_id] += self.weight_sparse * (
                1 / (self.k_const + rank)
            )

        # Sort by RRF score descending
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True
        )
        
        # Ensure top dense results are in final output
        final_ids = []
        for cid in top_dense_ids:
            if cid not in final_ids:
                final_ids.append(cid)
        
        # Add remaining by RRF score
        for cid in sorted_ids:
            if cid not in final_ids:
                final_ids.append(cid)

        if top_n_out is not None:
            final_ids = final_ids[:top_n_out]

        final_results = []
        for cid in final_ids:
            final_results.append((chunk_map[cid], rrf_scores[cid]))

        return final_results


if __name__ == "__main__":
    # Unit test for RRF calculation
    grouper = RRFGrouper(k_const=60)

    # Test data
    chunk_a = {"chunk_id": "A", "content": "A"}
    chunk_b = {"chunk_id": "B", "content": "B"}
    chunk_c = {"chunk_id": "C", "content": "C"}

    # Dense: A(rank 1), B(rank 2)
    dense = [(chunk_a, 0.9), (chunk_b, 0.8)]

    # Sparse: B(rank 1), C(rank 2)
    sparse = [(chunk_b, 10.0), (chunk_c, 5.0)]

    # Expected scores:
    # A: 1/(60+1) = 0.01639
    # B: 1/(60+2) + 1/(60+1) = 0.01612 + 0.01639 = 0.03251
    # C: 1/(60+2) = 0.01612
    # Expected order: B, A, C

    results = grouper.fuse(dense, sparse)

    print("RRF Results:")
    for chunk, score in results:
        print(f"{chunk['chunk_id']}: {score:.5f}")

    assert results[0][0]["chunk_id"] == "B", "RRF ordering incorrect"
    print("RRF Test Passed")
