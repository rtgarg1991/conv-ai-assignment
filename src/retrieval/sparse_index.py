import json
import pickle
import string
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem import PorterStemmer
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config


class SparseIndex:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.model_path = Config.DATA_DIR / "bm25_index.pkl"

        # Simple preprocessing
        # We need NLTK punkt for tokenization usually, but let's use a simple splitter to avoid downloading big NLTK data if possible,
        # or just ensure we downlaod it.
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        self.stemmer = PorterStemmer()

    def preprocess(self, text: str) -> List[str]:
        """Tokenizes and stems text."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Tokenize
        tokens = text.split()
        # Stem
        return [self.stemmer.stem(t) for t in tokens]

    def build_index(self, corpus_path: Path = Config.CORPUS_PATH):
        """Builds BM25 index from corpus."""
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")

        print("Loading corpus for Sparse Index...")
        with open(corpus_path, "r") as f:
            self.chunks = json.load(f)

        tokenized_corpus = []
        print(f"Tokenizing {len(self.chunks)} chunks...")
        for chunk in self.chunks:
            tokenized_corpus.append(self.preprocess(chunk["content"]))

        print("Building BM25 Index...")
        self.index = BM25Okapi(tokenized_corpus)

        # Save index + chunks reference
        # BM25 object is pickleable
        print(f"Saving index to {self.model_path}...")
        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "index": self.index,
                    "chunk_ids": [c["chunk_id"] for c in self.chunks],
                    # We don't save full chunks to save space, will reload corpus on load
                },
                f,
            )
        print("Sparse Index built.")

    def load_index(self):
        """Loads BM25 index."""
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
                self.index = data["index"]

            # Load chunks
            with open(Config.CORPUS_PATH, "r") as f:
                self.chunks = json.load(f)
        else:
            print("Sparse index not found. Please build it.")

    def search(self, query: str, k: int = 60) -> List[Tuple[Dict, float]]:
        """Retrieves top-k chunks using BM25."""
        if self.index is None:
            self.load_index()

        query_tokens = self.preprocess(query)
        doc_scores = self.index.get_scores(query_tokens)

        # Get top k indices
        # argsort returns ascending, so we take last k and reverse
        top_n = np.argsort(doc_scores)[-k:][::-1]

        results = []
        for idx in top_n:
            score = doc_scores[idx]
            # BM25 scores can be 0 if no closure match
            if score > 0:
                results.append((self.chunks[idx], float(score)))

        return results


import numpy as np  # Needed for argsort

if __name__ == "__main__":
    si = SparseIndex()

    if not si.model_path.exists():
        si.build_index()
    else:
        si.load_index()

    query = "philosophy of science"
    results = si.search(query, k=3)

    print(f"\nQuery: {query}")
    for chunk, score in results:
        print(f"[{score:.4f}] {chunk['title']}: {chunk['content'][:50]}...")
