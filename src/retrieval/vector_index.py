import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config


class VectorIndex:
    def __init__(self):
        self.model_name = Config.EMBEDDING_MODEL_NAME
        self.device = Config.DEVICE

        print(
            f"Loading embedding model: {self.model_name} on {self.device}..."
        )
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.index = None
        self.chunks = []  # Metadata store

    def build_index(self, corpus_path: Path = Config.CORPUS_PATH):
        """Loads corpus, encodes chunks, and builds FAISS index."""
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")

        print("Loading corpus...")
        with open(corpus_path, "r") as f:
            self.chunks = json.load(f)

        texts = [chunk["content"] for chunk in self.chunks]
        print(f"Encoding {len(texts)} chunks...")

        # Batch encoding
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Important for cosine similarity
        )

        # Create FAISS Index
        # IndexFlatIP (Inner Product) is equivalent to Cosine Similarity on normalized vectors
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Index built with {self.index.ntotal} vectors.")

        # Save index
        faiss.write_index(self.index, str(Config.VECTOR_DB_PATH))
        print(f"Index saved to {Config.VECTOR_DB_PATH}")

    def load_index(self):
        """Loads existing index and corpus metadata which matches it."""
        if Config.VECTOR_DB_PATH.exists():
            self.index = faiss.read_index(str(Config.VECTOR_DB_PATH))
            # creating parallel path for metadata since FAISS doesn't store it
            # In a real DB like Chroma/Pinecone, meatadata is attached. Here we just rely on order matching in corpus.json
            with open(Config.CORPUS_PATH, "r") as f:
                self.chunks = json.load(f)
        else:
            print("Index file not found. Please build it first.")

    def search(self, query: str, k: int = 60) -> List[Tuple[Dict, float]]:
        """Encodes query and retrieves top-k chunks with scores."""
        if self.index is None:
            self.load_index()

        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )

        distances, indices = self.index.search(query_vector, k)

        results = []
        for i in range(k):
            idx = indices[0][i]
            score = distances[0][i]
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results


if __name__ == "__main__":
    # Test
    vi = VectorIndex()

    # Build if needed, or just load
    if not Config.VECTOR_DB_PATH.exists():
        vi.build_index()
    else:
        vi.load_index()

    query = "What is the meaning of life?"
    results = vi.search(query, k=3)

    print(f"\nQuery: {query}")
    for chunk, score in results:
        print(f"[{score:.4f}] {chunk['title']}: {chunk['content'][:50]}...")
