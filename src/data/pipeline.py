import json
import time
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.config import Config
from src.data.url_loader import URLLoader
from src.data.scraping import Scraper
from src.data.chunking import Chunker


class DataPipeline:
    def __init__(self):
        self.loader = URLLoader()
        self.scraper = Scraper()
        self.chunker = Chunker(chunk_size=300, overlap=50)

    def run(self):
        print(f"--- Starting Data Pipeline ({Config.ENV} Mode) ---")

        # 1. Load URLs
        fixed_urls = self.loader.load_fixed_urls()
        random_urls = self.loader.load_random_urls(fixed_urls)

        all_urls = fixed_urls + random_urls
        print(f"Total URLs to process: {len(all_urls)}")

        # 2. Scrape & Chunk
        all_chunks = []
        documents_processed = 0

        for url in tqdm(all_urls, desc="Processing URLs"):
            # Scrape
            scrape_data = self.scraper.scrape_url(url)
            if not scrape_data:
                continue

            documents_processed += 1

            # Chunk
            chunks = self.chunker.chunk_text(
                scrape_data["content"], scrape_data
            )
            all_chunks.extend(chunks)

            # Rate limiting slightly to be polite
            time.sleep(0.1)

        # 3. Save Corpus
        print(f"\nProcessed {documents_processed}/{len(all_urls)} documents.")
        print(f"Generated {len(all_chunks)} chunks.")

        with open(Config.CORPUS_PATH, "w") as f:
            json.dump(all_chunks, f, indent=2)

        print(f"Corpus saved to {Config.CORPUS_PATH}")
        return len(all_chunks)


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
