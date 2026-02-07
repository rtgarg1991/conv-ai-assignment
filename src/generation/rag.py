"""
RAG (Retrieval-Augmented Generation) Service.

Orchestrates hybrid retrieval and LLM generation to answer user queries
using context from the knowledge base.
"""

from typing import List, Dict
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.generation.model_service import ModelService
from src.retrieval.engine import HybridRetriever


class RAGService:
    """
    Retrieval-Augmented Generation service.

    Combines hybrid retrieval with LLM generation to produce
    grounded answers from the knowledge base.
    """

    def __init__(self):
        """Initialize RAG service with retriever and model."""
        self.retriever = HybridRetriever()
        self.model_service = ModelService()
        self.is_t5 = self.model_service.is_t5

    def initialize(self):
        """Initialize retriever and model components."""
        print("Initializing RAG Service...")
        self.retriever.initialize()
        self.model_service.initialize()
        print("RAG Service Initialized.")

    def construct_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Construct generation prompt based on model type.

        Args:
            query: User question.
            context_chunks: Retrieved context chunks.

        Returns:
            Formatted prompt string.
        """
        # Limit context length to avoid exceeding model limits
        context_parts = []
        current_len = 0
        max_context_chars = Config.MAX_CONTEXT_CHARS

        for chunk in context_chunks:
            txt = chunk["content"]
            if current_len + len(txt) < max_context_chars:
                context_parts.append(txt)
                current_len += len(txt)
            else:
                break

        context_text = "\n\n".join(context_parts)

        if self.is_t5:
            # T5 instruction-following format
            prompt = f"Answer the question based on the context below.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
        else:
            # GPT-2 completion format
            prompt = f"""Use the following context to answer the question.
            
Context:
{context_text}

Question: {query}
Answer:"""

        return prompt

    def answer_question(self, query: str) -> Dict:
        """
        Run the full RAG pipeline: retrieve, construct prompt, generate.

        Args:
            query: User question.

        Returns:
            Dictionary with query, answer, and retrieved_chunks.
        """
        if self.retriever.vector_index.index is None:
            self.initialize()

        # Retrieve relevant context
        print(f"Retrieving context for: {query}")
        retrieved_chunks_scores = self.retriever.retrieve(query)
        retrieved_chunks = [c for c, s in retrieved_chunks_scores]

        # Construct prompt
        prompt = self.construct_prompt(query, retrieved_chunks)

        # Generate answer
        print("Generating answer...")
        answer = self.model_service.generate(prompt, max_new_tokens=150)

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks[:3],
        }

    def answer_question_with_details(self, query: str) -> Dict:
        """
        Run RAG pipeline with detailed retrieval scores for UI display.

        Returns:
            Dictionary with answer, chunks with individual scores, and timing.
        """
        import time

        if self.retriever.vector_index.index is None:
            self.initialize()

        # Retrieve with detailed scores
        print(f"Retrieving context for: {query}")
        retrieval_result = self.retriever.retrieve_with_details(query)
        enriched_chunks = retrieval_result["final_results"]

        # Construct prompt using basic chunk structure
        chunks_for_prompt = [
            {"content": c["content"], "title": c["title"]}
            for c in enriched_chunks
        ]
        prompt = self.construct_prompt(query, chunks_for_prompt)

        # Generate answer with timing
        print("Generating answer...")
        start = time.time()
        answer = self.model_service.generate(prompt, max_new_tokens=150)
        generation_ms = (time.time() - start) * 1000

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": enriched_chunks[:5],
            "timing": retrieval_result["timing"],
            "generation_ms": round(generation_ms, 2),
        }


if __name__ == "__main__":
    rag = RAGService()
    rag.initialize()

    query = "What is the central concept of Stoicism?"
    result = rag.answer_question(query)

    print(f"\nQUERY: {result['query']}")
    print(f"ANSWER: {result['answer']}")
    print("SOURCES:")
    for chunk in result["retrieved_chunks"]:
        print(f"- {chunk['title']}")
