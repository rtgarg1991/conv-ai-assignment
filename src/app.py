"""
Streamlit Web Application for Hybrid RAG System.

This module provides a web interface for querying the Hybrid RAG system
and running evaluation pipelines.
"""

import streamlit as st
import pandas as pd
import time
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import Config
from src.generation.rag import RAGService
from src.evaluation.runner import EvaluationRunner

# Page Configuration
st.set_page_config(
    page_title="Hybrid RAG System", page_icon="robot", layout="wide"
)

# Initialize Session State
if "rag_service" not in st.session_state:
    st.session_state.rag_service = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


@st.cache_resource
def get_rag_service():
    """Initialize and cache RAG Service instance."""
    service = RAGService()
    service.initialize()
    return service


def main():
    """Main application entry point."""
    st.title("Hybrid RAG System")
    st.markdown(
        f"Running on **{Config.DEVICE}** with Generator **{Config.GENERATION_MODEL}**"
    )

    # Sidebar for Administration and Evaluation Controls
    with st.sidebar:
        st.header("System Controls")
        if st.button("Reload System"):
            st.cache_resource.clear()
            st.success(
                "Cache cleared. RAG Service will reload on next query."
            )

        st.divider()
        st.subheader("Evaluation Pipeline")
        if st.button("Run Evaluation"):
            with st.spinner(
                "Running evaluation pipeline. This may take several minutes..."
            ):
                runner = EvaluationRunner()
                runner.run_evaluation()
            st.success(
                "Evaluation complete. Results saved to `data/evaluation_results.csv`."
            )

            # Display results if available
            eval_path = Config.DATA_DIR / "evaluation_results.csv"
            if eval_path.exists():
                st.dataframe(pd.read_csv(eval_path))

    # Main Query Interface
    try:
        rag_service = get_rag_service()
    except Exception as e:
        st.error(f"Failed to initialize RAG Service: {e}")
        return

    # User Input
    user_query = st.text_input(
        "Ask a question about the corpus:",
        placeholder="e.g., What is the central concept of Stoicism?",
    )

    if user_query:
        with st.spinner("Processing query..."):
            start_time = time.time()
            result = rag_service.answer_question_with_details(user_query)
            total_latency = time.time() - start_time

        # Display Answer
        st.success(f"**Answer:** {result['answer']}")

        # Latency breakdown
        timing = result.get("timing", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", f"{total_latency:.2f}s")
        col2.metric("Dense", f"{timing.get('dense_ms', 0):.0f}ms")
        col3.metric("Sparse", f"{timing.get('sparse_ms', 0):.0f}ms")
        col4.metric("Generation", f"{result.get('generation_ms', 0):.0f}ms")

        # Display Retrieved Context with individual scores
        with st.expander("🔍 Retrieved Context (Hybrid RRF)", expanded=True):
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                # Score badges
                dense_score = chunk.get("dense_score", 0)
                sparse_score = chunk.get("sparse_score", 0)
                rrf_score = chunk.get("rrf_score", 0)

                st.markdown(f"### {i}. {chunk['title']}")

                # Score display in columns
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Dense", f"{dense_score:.4f}")
                sc2.metric("Sparse", f"{sparse_score:.2f}")
                sc3.metric("RRF", f"{rrf_score:.4f}")

                st.info(
                    chunk["content"][:500] + "..."
                    if len(chunk["content"]) > 500
                    else chunk["content"]
                )
                st.markdown(f"[📄 Source]({chunk['url']})")
                st.divider()

    # Footer
    st.divider()
    st.caption("Hybrid RAG System | Dense + Sparse + RRF | Assignment 2")


if __name__ == "__main__":
    main()
