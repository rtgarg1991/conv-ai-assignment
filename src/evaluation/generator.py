import json
import random
import sys
import os
from typing import List, Dict
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

    def load_corpus(self):
        with open(self.corpus_path, "r") as f:
            self.chunks = json.load(f)

    def generate_dataset(self, num_samples: int = 10):
        """Generates Q&A pairs from random chunks."""
        self.load_corpus()

        # Select random chunks
        selected_chunks = random.sample(
            self.chunks, min(num_samples, len(self.chunks))
        )

        qa_dataset = []

        print(f"Generating {len(selected_chunks)} Q&A pairs...")
        self.model_service.initialize()

        for chunk in tqdm.tqdm(selected_chunks):
            qa_pair = self.generate_single_qa(chunk)
            if qa_pair:
                qa_dataset.append(qa_pair)

        # Save
        print(f"Saving {len(qa_dataset)} pairs to {self.output_path}...")
        with open(self.output_path, "w") as f:
            json.dump(qa_dataset, f, indent=4)

    def generate_single_qa(self, chunk: Dict) -> Dict:
        """Generates a Q&A pair for a single chunk."""
        context = chunk["content"]

        # Prompt Strategy depends on model
        # For GPT2 (Causal), we need to guide it to complete a pattern.
        # For T5, we can give a direct instruction.

        if self.model_service.is_t5:
            prompt = f"Generate a question based on this context: {context}\n\nQuestion:"
            question = self.model_service.generate(prompt, max_length=64)

            # Answer is actually the context itself for retrieval evaluation usually,
            # or we can ask it to generate an answer too.
            # For MRR, we just need the Question and the Ground Truth URL/Chunk ID.
            # But let's try to get an answer text for ROUGE eval.

            prompt_ans = f"Context: {context}\nQuestion: {question}\nAnswer:"
            answer = self.model_service.generate(prompt_ans, max_length=64)

        else:
            # GPT2 Strategy: Few-shot completion style might be safer
            # But given the small context window and model size, let's try a direct completion
            # "Context: ... \n\nGenerate a question for the above text: "

            prompt = f"""Context: {context[:1000]}
            
Task: Write a question about the text above.
Question:"""
            question = self.model_service.generate(prompt, max_new_tokens=40)

            # Clean up: GPT2 might generate more context or repeat
            # Take first line or sentence
            question = question.split("\n")[0].strip()

            # Use the chunk content as the reference answer for now,
            # or generate one? Generating one with small GPT2 might be circular.
            # Let's use the chunk content (snippet) as the "Ground Truth Info"
            answer = context[:200]

        return {
            "question": question,
            "answer": answer,  # Generated answer or snippet
            "chunk_id": chunk["chunk_id"],
            "url": chunk["url"],
            "ground_truth_context": chunk["content"],
        }


if __name__ == "__main__":
    gen = QAGenerator()
    # Generate a small set first
    gen.generate_dataset(num_samples=5)
