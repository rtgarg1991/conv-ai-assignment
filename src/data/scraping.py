import re
import wikipediaapi
from bs4 import BeautifulSoup
import sys
import os
import requests
from urllib.parse import unquote


class Scraper:
    def __init__(self):
        # We use wikipedia-api for easy metadata, but sometimes we might need raw HTML if the API suppresses too much.
        # Actually, wikipedia-api gives clean text usually.
        # Assignment asks to "Extract, clean... remove Edit buttons, references".
        # wikipedia-api .text property is usually decent, but let's ensure we strip artifacts.
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="HybridRAG_Assignment/1.0 (contact@example.com)",
            language="en",
        )

    def clean_text(self, text: str) -> str:
        """Cleans raw text by removing citations, extra whitespace, etc."""
        # Remove [1], [15], cast [citation needed]
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\[citation needed\]", "", text)

        # Remove "Edit" markings if they slipped through (common in some raw scrapes, less in api)
        text = re.sub(r"\bEdit\b", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def scrape_url(self, url: str) -> dict:
        """Fetches and cleans text from a Wikipedia URL."""
        try:
            # Extract and properly decode title from URL
            raw_title = url.split("/wiki/")[-1]
            title = unquote(raw_title).replace("_", " ")

            page = self.wiki.page(title)
            if not page.exists():
                print(f"Page not found: {title}")
                return None

            # Helper to get full text including sections, but skipping References/External Links
            # wikipedia-api properties:
            # page.text: Gives full text. Usually skips References section content but includes header.
            # Let's clean it.

            raw_text = page.text

            # Simple heuristic to cut off footer sections if they appear in .text
            # Usually "References", "See also", "External links" might be at the end.
            # But .text usually handles structure well.
            # We'll just run our cleaner.

            cleaned_text = self.clean_text(raw_text)

            # Validate length (>200 words)
            word_count = len(cleaned_text.split())
            if word_count < 200:
                print(f"Skipping {title}: Too short ({word_count} words)")
                return None

            return {
                "url": url,
                "title": page.title,
                "content": cleaned_text,
                "summary": self.clean_text(page.summary),
            }

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None


if __name__ == "__main__":
    # Test Scraper
    scraper = Scraper()
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    result = scraper.scrape_url(test_url)

    if result:
        print(f"Title: {result['title']}")
        print(f"Content Length: {len(result['content'])} chars")
        print(f"Sample: {result['content'][:200]}...")
        if "[1]" in result["content"]:
            print("❌ Failed to remove citations")
        else:
            print("✅ Citations removed")
    else:
        print("Scraping failed")
