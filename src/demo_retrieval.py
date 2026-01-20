import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.retrieval.engine import HybridRetriever


def main():
    print("Initializing Hybrid Retrieval Engine...")
    retriever = HybridRetriever()
    retriever.initialize()

    queries = [
        "What is the central concept of Stoicism?",
        "Explain the scientific method",
        "Who is Aristotle?",
    ]

    print("\n" + "=" * 50)
    print("       HYBRID RETRIEVAL DEMO       ")
    print("=" * 50)

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)

        try:
            results = retriever.retrieve(query)

            for i, (chunk, score) in enumerate(results, 1):
                # Clean up newlines for display
                snippet = chunk["content"][:150].replace("\n", " ")
                print(f"{i}. [Score: {score:.4f}] {chunk['title']}")
                print(f'   "{snippet}..."')
                print()
        except Exception as e:
            print(f"Error retrieving for '{query}': {e}")

    print("=" * 50)


if __name__ == "__main__":
    main()
