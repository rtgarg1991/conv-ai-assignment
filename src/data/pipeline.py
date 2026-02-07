import json
import time
from tqdm import tqdm

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

        # Deduplicate URLs (safety check)
        all_urls = list(
            dict.fromkeys(fixed_urls + random_urls)
        )  # Preserves order, removes dupes
        if len(all_urls) < len(fixed_urls) + len(random_urls):
            print(
                f"⚠️ Removed {len(fixed_urls) + len(random_urls) - len(all_urls)} duplicate URLs"
            )
        print(f"Total URLs to process: {len(all_urls)}")

        # 2. Scrape & Chunk
        all_chunks = []
        documents_processed = 0
        fixed_count, random_count = Config.get_url_counts()
        target_urls = fixed_count + random_count

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

        # 3. Check if we have enough URLs, retry if needed
        processed_urls = set(c["url"] for c in all_chunks)
        retry_count = 0
        max_retries = 10
        while len(processed_urls) < target_urls and retry_count < max_retries:
            needed = target_urls - len(processed_urls)
            print(
                f"\nNeed {needed} more URLs, fetching additional random URLs..."
            )
            extra_urls = self.loader.load_random_urls(
                list(processed_urls), count=needed + 5
            )

            for url in tqdm(extra_urls, desc="Processing extra URLs"):
                if url in processed_urls:
                    continue
                scrape_data = self.scraper.scrape_url(url)
                if not scrape_data:
                    continue
                chunks = self.chunker.chunk_text(
                    scrape_data["content"], scrape_data
                )
                if chunks:
                    all_chunks.extend(chunks)
                    processed_urls.add(url)
                    documents_processed += 1
                time.sleep(0.1)
            retry_count += 1

        # 4. Save Corpus
        print(f"\nProcessed {len(processed_urls)}/{target_urls} documents.")
        print(f"Generated {len(all_chunks)} chunks.")

        with open(Config.CORPUS_PATH, "w") as f:
            json.dump(all_chunks, f, indent=2)

        print(f"Corpus saved to {Config.CORPUS_PATH}")
        return len(all_chunks)


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
