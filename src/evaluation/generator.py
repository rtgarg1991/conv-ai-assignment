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


class QAGenerator:
    def __init__(self):
        self.model_service = ModelService()
        self.corpus_path = Config.CORPUS_PATH
        self.output_path = Config.DATA_DIR / "qa_dataset.json"
        self.min_question_length = 10  # Filter out too-short questions

    def load_corpus(self):
        with open(self.corpus_path, "r") as f:
            self.chunks = json.load(f)

    def generate_dataset(self, num_samples: int = 100):
        """Generates Q&A pairs from diverse chunks across different articles."""
        self.load_corpus()

        # Group chunks by URL for diversity
        chunks_by_url = defaultdict(list)
        for chunk in self.chunks:
            chunks_by_url[chunk["url"]].append(chunk)
        
        # Select chunks ensuring diversity across articles
        selected_chunks = self._select_diverse_chunks(chunks_by_url, num_samples)
        
        qa_dataset = []
        failed_count = 0

        print(f"Generating {len(selected_chunks)} Q&A pairs from {len(set(c['url'] for c in selected_chunks))} articles...")
        self.model_service.initialize()

        for chunk in tqdm.tqdm(selected_chunks, desc="Generating Q&A"):
            try:
                qa_pair = self.generate_single_qa(chunk)
                if qa_pair and self._is_quality_qa(qa_pair):
                    qa_dataset.append(qa_pair)
                else:
                    failed_count += 1
            except Exception as e:
                print(f"\nError generating Q&A: {e}")
                failed_count += 1

        print(f"\nGenerated {len(qa_dataset)} valid Q&A pairs ({failed_count} failed/filtered)")
        
        # Save
        print(f"Saving to {self.output_path}...")
        with open(self.output_path, "w") as f:
            json.dump(qa_dataset, f, indent=2)

    def _select_diverse_chunks(self, chunks_by_url: Dict, num_samples: int) -> List[Dict]:
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
        question_indicators = ["?", "what", "who", "where", "when", "why", "how", "which", "is ", "are ", "was ", "were ", "do ", "does ", "did "]
        has_indicator = any(ind in question.lower() for ind in question_indicators)
        if not has_indicator:
            return False
        
        return True

    def generate_single_qa(self, chunk: Dict) -> Dict:
        """Generates a Q&A pair for a single chunk."""
        context = chunk["content"][:1500]  # Limit context size
        title = chunk.get("title", "Unknown")

        if self.model_service.is_t5:
            # Better prompt for T5 - more specific instruction
            prompt = f"""Generate a factual question that can be answered using this text about {title}:

{context[:800]}

Question:"""
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
            # GPT2 Strategy
            prompt = f"""Text: {context[:800]}

Write a question about the text above:
Question:"""
            question = self.model_service.generate(prompt, max_new_tokens=50)
            question = question.split("\n")[0].strip()
            
            if question and not question.endswith("?"):
                question += "?"
            
            # Use context snippet as answer for GPT2
            answer = context[:300]

        return {
            "question": question,
            "answer": answer,
            "chunk_id": chunk["chunk_id"],
            "url": chunk["url"],
            "title": title,
            "ground_truth_context": chunk["content"],
        }


if __name__ == "__main__":
    gen = QAGenerator()
    # Generate 100 Q&A pairs as required by assignment
    gen.generate_dataset(num_samples=100)
