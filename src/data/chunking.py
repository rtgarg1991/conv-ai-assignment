import uuid
from typing import List, Dict


class Chunker:
    def __init__(self, chunk_size=300, overlap=50, min_chunk_size=200):
        # Assignment asks for 200-400 tokens with 50-token overlap.
        # We'll default to 300.
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size  # Enforce 200+ token chunks

    def chunk_text(self, text: str, meta: Dict) -> List[Dict]:
        """Splits text into overlapping chunks with metadata."""
        # Simple whitespace tokenization (sufficient for assignment requirements)
        # For more precision, we could use nltk or a bert tokenizer, but split() is robust for general purpose.
        tokens = text.split()

        chunks = []

        if not tokens:
            return []

        # If text is shorter than min chunk size, drop it (assignment requires 200+ tokens)
        if len(tokens) < self.min_chunk_size:
            return []

        # If text is shorter than chunk size, take it all (>= min_chunk_size guaranteed)
        if len(tokens) <= self.chunk_size:
            chunk_text = " ".join(tokens)
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "url": meta["url"],
                    "title": meta["title"],
                    "content": chunk_text,
                    "token_count": len(tokens),
                }
            )
            return chunks

        # Rolling window
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)

            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "url": meta["url"],
                    "title": meta["title"],
                    "content": chunk_text,
                    "token_count": len(chunk_tokens),
                }
            )

            # Stop if we reached the end
            if end >= len(tokens):
                break

            # Move forward by (chunk_size - overlap)
            start += self.chunk_size - self.overlap

        # Filter out chunks that are too small (tail chunks)
        filtered_chunks = [
            c for c in chunks if c["token_count"] >= self.min_chunk_size
        ]
        return filtered_chunks


if __name__ == "__main__":
    # Test Chunking
    chunker = Chunker(chunk_size=10, overlap=2)
    dummy_text = (
        "one two three four five six seven eight nine ten eleven twelve"
    )
    meta = {"url": "http://test", "title": "Test Doc"}

    result = chunker.chunk_text(dummy_text, meta)

    print(f"Original: {dummy_text}")
    print(f"Chunk Config: Size=10, Overlap=2")
    print(f"Chunks Generated: {len(result)}")
    for i, c in enumerate(result):
        print(f"Chunk {i}: {c['content']} (Len: {c['token_count']})")

    # Validation check
    # Chunk 0: 1..10
    # Chunk 1: (10-2)+1 = 9..18 -> "nine ten eleven twelve"
    # Wait, simple math check:
    # Start=0, End=10. Tokens: 1..10. Next Start = 0 + (10-2) = 8.
    # Next tokens index 8 is "nine" (0-indexed: 0=one... 8=nine).
    # Correct.
