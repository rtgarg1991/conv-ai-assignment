"""
Configuration module for Hybrid RAG System.

Handles environment detection, device selection, and system parameters.
Environment variable RAG_ENV controls dataset size (LOCAL or PROD).
"""

import os
import torch
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


class Config:
    """Central configuration for the Hybrid RAG system."""

    # Environment
    ENV = os.getenv("RAG_ENV", "LOCAL").upper()

    # Device (M1 Support)
    if torch.backends.mps.is_available():
        DEVICE = "mps"
    elif torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    # Paths
    DATA_DIR = DATA_DIR
    FIXED_URLS_PATH = DATA_DIR / "fixed_urls.json"
    CORPUS_PATH = DATA_DIR / "corpus.json"
    VECTOR_DB_PATH = DATA_DIR / "vector_index.faiss"

    # Models
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Generation Model Options: "gpt2", "gpt2-medium", "google/flan-t5-base", "google/flan-t5-small"
    GENERATION_MODEL = "gpt2"

    # Legacy LLM name for backward compatibility
    LLM_MODEL_NAME = "google/flan-t5-base"

    # Retrieval Parameters
    RRF_K = 60
    TOP_N_RETRIEVAL = 5
    MAX_CONTEXT_CHARS = 2000

    @classmethod
    def get_url_counts(cls):
        """Return (fixed_count, random_count) based on environment."""
        if cls.ENV == "PROD":
            return 200, 300
        else:
            # LOCAL / DEV
            return 50, 100

    @classmethod
    def __repr__(cls):
        fixed, random = cls.get_url_counts()
        return (
            f"Config(ENV={cls.ENV}, DEVICE={cls.DEVICE}, "
            f"URLS=Fixed:{fixed}+Random:{random}, "
            f"MODELS={cls.EMBEDDING_MODEL_NAME} & {cls.LLM_MODEL_NAME})"
        )
