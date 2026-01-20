from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config


class ModelService:
    def __init__(self):
        self.model_name = Config.GENERATION_MODEL
        self.device = Config.DEVICE
        self.tokenizer = None
        self.model = None
        self.is_t5 = "t5" in self.model_name.lower()

    def initialize(self):
        """Loads model and tokenizer based on config."""
        print(f"Loading GenAI Model: {self.model_name} on {self.device}...")

        if self.is_t5:
            # T5 Loading
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name, legacy=False
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name
            ).to(self.device)
        else:
            # GPT2 Loading (Causal LM)
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(
                self.device
            )
            # GPT2 doesn't have a pad token by default, set it to eos
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Model loaded.")

    def generate(
        self, prompt: str, max_length: int = 200, max_new_tokens: int = None
    ) -> str:
        """Generates text from prompt."""
        if self.model is None:
            self.initialize()

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.device
        )

        # Calculate max_length if not provided but max_new_tokens is
        generated_kwargs = {}
        if max_new_tokens:
            generated_kwargs["max_new_tokens"] = max_new_tokens
        else:
            generated_kwargs["max_length"] = max_length

        if self.is_t5:
            # Seq2Seq Generation
            outputs = self.model.generate(
                input_ids,
                num_beams=5,
                early_stopping=True,
                **generated_kwargs,
            )
        else:
            # Causal LM Generation
            outputs = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                # Sampling params for better creativity/coherence in GPT2
                do_sample=True,
                top_k=50,
                top_p=0.95,
                **generated_kwargs,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    ms = ModelService()
    ms.initialize()

    if ms.is_t5:
        prompt = (
            "Answer the following question: What is the capital of France?"
        )
    else:
        prompt = "The capital of France is"

    print(f"\nPrompt: {prompt}")
    answer = ms.generate(prompt)
    print(f"Answer: {answer}")
