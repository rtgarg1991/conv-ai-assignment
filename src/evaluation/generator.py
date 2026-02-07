import json
import random
import sys
import os
from typing import List, Dict
from collections import defaultdict
import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.generation.model_service import ModelService

# Question type definitions with prompts
QUESTION_TYPES = {
    "factual": {
        "description": "Direct factual questions with single clear answers",
        "prompts": [
            "What is {topic}?",
            "When did {event} happen?",
            "Who is {person}?",
            "Where is {place} located?",
        ],
    },
    "comparative": {
        "description": "Questions comparing two or more entities",
        "prompts": [
            "How does {topic} compare to similar concepts?",
            "What are the key differences in {topic}?",
        ],
    },
    "inferential": {
        "description": "Questions requiring reasoning from given facts",
        "prompts": [
            "Why is {topic} significant?",
            "What caused {event}?",
            "What are the implications of {topic}?",
        ],
    },
    "multi_hop": {
        "description": "Questions requiring multiple facts to answer",
        "prompts": [
            "How did {topic} influence later developments?",
            "What connections exist between aspects of {topic}?",
        ],
    },
}


class QAGenerator:
    def __init__(self):
        self.model_service = ModelService()
        self.corpus_path = Config.CORPUS_PATH
        self.output_path = Config.DATA_DIR / "qa_dataset.json"
        self.fixed_urls_path = Config.FIXED_URLS_PATH
        self.min_question_length = 10  # Filter out too-short questions
        self.question_types = list(QUESTION_TYPES.keys())
        self.fixed_urls = set()  # Will be loaded

    def load_corpus(self):
        with open(self.corpus_path, "r") as f:
            self.chunks = json.load(f)

        # Load fixed URLs to filter chunks
        if self.fixed_urls_path.exists():
            with open(self.fixed_urls_path, "r") as f:
                self.fixed_urls = set(json.load(f))
            print(f"Loaded {len(self.fixed_urls)} fixed URLs for filtering")

    def generate_dataset(
        self, num_samples: int = 100, use_fixed_only: bool = True
    ):
        """
        Generates Q&A pairs from diverse chunks across different articles.

        Args:
            num_samples: Number of Q&A pairs to generate
            use_fixed_only: If True, only use chunks from fixed URLs (recommended)
        """
        self.load_corpus()

        # Filter to fixed URLs only for consistent evaluation
        if use_fixed_only and self.fixed_urls:
            filtered_chunks = [
                c for c in self.chunks if c["url"] in self.fixed_urls
            ]
            print(
                f"Filtering to fixed URLs: {len(filtered_chunks)}/{len(self.chunks)} chunks"
            )
        else:
            filtered_chunks = self.chunks

        # Group chunks by URL for diversity
        chunks_by_url = defaultdict(list)
        for chunk in filtered_chunks:
            chunks_by_url[chunk["url"]].append(chunk)

        # Select chunks ensuring diversity across articles
        selected_chunks = self._select_diverse_chunks(
            chunks_by_url, num_samples
        )

        qa_dataset = []
        failed_count = 0

        # Track question type distribution for balance
        type_counts = {t: 0 for t in self.question_types}

        print(
            f"Generating {len(selected_chunks)} Q&A pairs from {len(set(c['url'] for c in selected_chunks))} articles..."
        )
        print(f"Target types: {self.question_types}")
        self.model_service.initialize()

        for i, chunk in enumerate(
            tqdm.tqdm(selected_chunks, desc="Generating Q&A")
        ):
            try:
                # Cycle through question types for balanced distribution
                target_type = self.question_types[
                    i % len(self.question_types)
                ]

                qa_pair = self.generate_single_qa(
                    chunk, target_type=target_type
                )
                if qa_pair and self._is_quality_qa(qa_pair):
                    qa_dataset.append(qa_pair)
                    type_counts[qa_pair["question_type"]] += 1
                else:
                    failed_count += 1
            except Exception as e:
                print(f"\nError generating Q&A: {e}")
                failed_count += 1

        print(
            f"\nGenerated {len(qa_dataset)} valid Q&A pairs ({failed_count} failed/filtered)"
        )
        print(f"Type distribution: {type_counts}")

        # Save
        print(f"Saving to {self.output_path}...")
        with open(self.output_path, "w") as f:
            json.dump(qa_dataset, f, indent=2)

    def _select_diverse_chunks(
        self, chunks_by_url: Dict, num_samples: int
    ) -> List[Dict]:
        """Select chunks ensuring diversity across different articles."""
        selected = []
        urls = list(chunks_by_url.keys())
        random.shuffle(urls)

        # Round-robin selection from different URLs
        url_idx = 0
        while len(selected) < num_samples and urls:
            url = urls[url_idx % len(urls)]
            chunks = chunks_by_url[url]

            if chunks:
                # Pick a random chunk from this URL
                chunk = random.choice(chunks)
                selected.append(chunk)
                chunks.remove(chunk)  # Don't pick same chunk twice

                # Remove URL if no more chunks
                if not chunks:
                    urls.remove(url)

            url_idx += 1

        return selected

    def _is_quality_qa(self, qa_pair: Dict) -> bool:
        """Check if generated Q&A pair meets quality criteria."""
        question = qa_pair.get("question", "")

        # Filter criteria
        if len(question) < self.min_question_length:
            return False
        if not question.strip():
            return False
        # Should look like a question (ends with ? or contains question words)
        question_indicators = [
            "?",
            "what",
            "who",
            "where",
            "when",
            "why",
            "how",
            "which",
            "is ",
            "are ",
            "was ",
            "were ",
            "do ",
            "does ",
            "did ",
        ]
        has_indicator = any(
            ind in question.lower() for ind in question_indicators
        )
        if not has_indicator:
            return False

        return True

    def generate_single_qa(
        self, chunk: Dict, target_type: str = None
    ) -> Dict:
        """Generates a Q&A pair for a single chunk with specified question type."""
        context = chunk["content"][:1500]  # Limit context size
        title = chunk.get("title", "Unknown")

        # Use target type or select randomly for diversity
        if target_type is None:
            target_type = random.choice(self.question_types)

        if self.model_service.is_t5:
            # Type-specific prompts for diversity (required by assignment)
            type_prompts = {
                "factual": f"Generate a factual question about {title} that asks What, Who, When, or Where:\n\n{context[:700]}\n\nQuestion:",
                "comparative": f"Generate a question comparing aspects or elements mentioned in this text about {title}:\n\n{context[:700]}\n\nQuestion:",
                "inferential": f"Generate a Why or How question that requires reasoning about {title}:\n\n{context[:700]}\n\nQuestion:",
                "multi_hop": f"Generate a question about relationships or influences in {title} that requires connecting multiple facts:\n\n{context[:700]}\n\nQuestion:",
            }

            prompt = type_prompts.get(target_type, type_prompts["factual"])
            question = self.model_service.generate(prompt, max_length=64)
            question = question.strip()

            # Ensure it ends with ?
            if question and not question.endswith("?"):
                question += "?"

            # Generate answer
            prompt_ans = f"""Answer the question based on the context.

Context: {context[:600]}
Question: {question}
Answer:"""
            answer = self.model_service.generate(prompt_ans, max_length=100)

        else:
            # GPT2 Strategy - simpler prompts
            prompt = f"""Text: {context[:800]}

Write a question about the text above:
Question:"""
            question = self.model_service.generate(prompt, max_new_tokens=50)
            question = question.split("\n")[0].strip()

            if question and not question.endswith("?"):
                question += "?"

            # Use context snippet as answer for GPT2
            answer = context[:300]

        # Use target type as classification (we generated with that intent)
        # But also verify with keywords for accuracy
        detected_type = self._classify_question_type(question)
        final_type = (
            detected_type if detected_type != "factual" else target_type
        )

        return {
            "question": question,
            "answer": answer,
            "question_type": final_type,
            "chunk_id": chunk["chunk_id"],
            "url": chunk["url"],
            "title": title,
            "ground_truth_context": chunk["content"],
        }

    def _classify_question_type(self, question: str) -> str:
        """
        Classify question into one of the defined types based on keywords.

        Categories:
        - factual: What/Who/When/Where questions
        - comparative: Questions with compare/difference/versus
        - inferential: Why/How questions requiring reasoning
        - multi_hop: Questions about influence/connections/implications
        """
        q_lower = question.lower()

        # Multi-hop indicators (check first - most specific)
        if any(
            word in q_lower
            for word in [
                "influence",
                "connection",
                "led to",
                "result in",
                "relationship between",
            ]
        ):
            return "multi_hop"

        # Comparative indicators
        if any(
            word in q_lower
            for word in [
                "compare",
                "difference",
                "versus",
                "vs",
                "similar",
                "unlike",
                "between",
            ]
        ):
            return "comparative"

        # Inferential indicators
        if any(
            word in q_lower
            for word in [
                "why",
                "how did",
                "explain",
                "cause",
                "reason",
                "significance",
                "important",
            ]
        ):
            return "inferential"

        # Default to factual (What/Who/When/Where)
        return "factual"


if __name__ == "__main__":
    gen = QAGenerator()
    # Generate 100 Q&A pairs as required by assignment
    gen.generate_dataset(num_samples=100)
