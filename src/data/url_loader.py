"""
Wikipedia URL Loader module.

Fetches Wikipedia article URLs using multiple strategies:
1. Category crawling for diversity
2. Wikipedia Random API for volume
3. Pre-validation to ensure article quality
"""

import json
import random
import wikipediaapi
import requests
from pathlib import Path
from typing import List, Set
import sys
import os
import time

# Add src to path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config


class URLLoader:
    """
    Wikipedia URL loader with multiple fetching strategies.

    Combines category-based crawling with API-based random article
    fetching to ensure sufficient high-quality articles.
    """

    def __init__(self):
        """Initialize URL loader with Wikipedia API client."""
        self.wiki = wikipediaapi.Wikipedia(
            user_agent="HybridRAG_Assignment/1.0 (contact@example.com)",
            language="en",
        )
        self.fixed_path = Config.FIXED_URLS_PATH
        self.fixed_count, self.random_count = Config.get_url_counts()
        self.session = requests.Session()

    def get_random_article_via_api(self) -> str:
        """
        Fetch a random Wikipedia article URL using the MediaWiki API.

        Returns:
            Article URL or None if fetch failed.
        """
        try:
            response = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": 0,  # Main namespace only
                    "rnlimit": 1,
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                if "query" in data and "random" in data["query"]:
                    title = data["query"]["random"][0]["title"]
                    return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        except Exception as e:
            print(f"API fetch error: {e}")

        return None

    def validate_article(self, url: str) -> bool:
        """
        Pre-validate article to ensure it meets minimum requirements.

        Args:
            url: Wikipedia article URL.

        Returns:
            True if article is valid (exists and has sufficient content).
        """
        try:
            title = url.split("/wiki/")[-1].replace("_", " ")
            page = self.wiki.page(title)

            if not page.exists():
                return False

            # Check word count
            word_count = len(page.text.split())
            return word_count >= 200

        except Exception:
            return False

    def get_random_pages(
        self, count: int, exclude_titles: Set[str] = None
    ) -> List[str]:
        """
        Fetch random Wikipedia article URLs using multiple strategies.

        Uses a hybrid approach:
        1. Category crawling for diversity (up to 30% of requested)
        2. Wikipedia Random API for volume (remaining)

        Args:
            count: Number of URLs to fetch.
            exclude_titles: Set of titles to exclude.

        Returns:
            List of valid Wikipedia article URLs.
        """
        pages = []
        if exclude_titles is None:
            exclude_titles = set()

        print(f"Fetching {count} random pages...")

        # Strategy 1: Category-based (30% for diversity)
        category_target = min(int(count * 0.3), 100)
        pages.extend(
            self._fetch_from_categories(category_target, exclude_titles)
        )

        # Strategy 2: Random API (remaining)
        api_target = count - len(pages)
        pages.extend(
            self._fetch_via_random_api(api_target, exclude_titles, pages)
        )

        print(f"\nSuccessfully fetched {len(pages)}/{count} valid articles.")
        return pages[:count]

    def _fetch_from_categories(
        self, target: int, exclude_titles: Set[str]
    ) -> List[str]:
        """Fetch articles from diverse Wikipedia categories."""
        # Expanded category list for better coverage
        categories = [
            "History",
            "Physics",
            "Technology",
            "Biology",
            "Philosophy",
            "Art",
            "Literature",
            "Geography",
            "Mathematics",
            "Chemistry",
            "Astronomy",
            "Politics",
            "Economics",
            "Psychology",
            "Music",
            "Architecture",
            "Medicine",
            "Computer_science",
            "Linguistics",
            "Anthropology",
            "Sociology",
            "Law",
            "Engineering",
        ]
        random.shuffle(categories)

        pages = []
        gathered_titles = set()

        def fetch_from_category(cat_name, limit):
            """Recursively fetch pages from a category."""
            try:
                cat = self.wiki.page(f"Category:{cat_name}")
                if not cat.exists():
                    return

                for member in cat.categorymembers.values():
                    if len(pages) >= limit:
                        return

                    # Only process main namespace articles
                    if (
                        member.ns == wikipediaapi.Namespace.MAIN
                        and member.title not in exclude_titles
                        and member.title not in gathered_titles
                    ):
                        url = member.fullurl

                        # Pre-validate to avoid wasting slots
                        if self.validate_article(url):
                            pages.append(url)
                            gathered_titles.add(member.title)
                            print(
                                f"Category: {len(pages)}/{target}", end="\r"
                            )

            except Exception as e:
                pass  # Skip problematic categories

        # Fetch from categories
        for cat in categories:
            if len(pages) >= target:
                break
            fetch_from_category(cat, target)

        return pages

    def _fetch_via_random_api(
        self, target: int, exclude_titles: Set[str], existing_pages: List[str]
    ) -> List[str]:
        """Fetch random articles using Wikipedia's Random API."""
        pages = []
        existing_urls = set(existing_pages)
        attempts = 0
        max_attempts = target * 3  # Allow some retries for duplicates/invalid

        print(f"\nFetching {target} articles via Random API...")

        while len(pages) < target and attempts < max_attempts:
            url = self.get_random_article_via_api()

            if url and url not in existing_urls:
                # Pre-validate
                if self.validate_article(url):
                    pages.append(url)
                    existing_urls.add(url)
                    print(f"Random API: {len(pages)}/{target}", end="\r")

            attempts += 1
            time.sleep(0.05)  # Rate limiting

        return pages

    def load_fixed_urls(self) -> List[str]:
        """
        Load fixed URLs from JSON, or generate/save them if missing.

        Returns:
            List of fixed Wikipedia URLs.
        """
        if self.fixed_path.exists():
            print(f"Loading existing fixed URLs from {self.fixed_path}")
            with open(self.fixed_path, "r") as f:
                urls = json.load(f)
                if len(urls) >= self.fixed_count:
                    return urls[: self.fixed_count]
                print(
                    f"Warning: Found {len(urls)} URLs, but need {self.fixed_count}. Fetching more."
                )

        # Generate new fixed set
        print("Generating new fixed URL set...")
        urls = self.get_random_pages(self.fixed_count)

        # Save
        with open(self.fixed_path, "w") as f:
            json.dump(urls, f, indent=2)
        print(f"Saved {len(urls)} fixed URLs to {self.fixed_path}")
        return urls

    def load_random_urls(self, existing_urls: List[str]) -> List[str]:
        """
        Generate random URLs, avoiding overlaps with fixed set.

        Args:
            existing_urls: URLs to exclude (e.g., fixed URLs).

        Returns:
            List of random Wikipedia URLs.
        """
        # Extract titles from existing URLs
        exclude_titles = set()
        for url in existing_urls:
            try:
                title = url.split("/wiki/")[-1].replace("_", " ")
                exclude_titles.add(title)
            except:
                pass

        return self.get_random_pages(self.random_count, exclude_titles)


if __name__ == "__main__":
    loader = URLLoader()
    fixed = loader.load_fixed_urls()
    print(f"\nFixed URLs: {len(fixed)}")

    random_set = loader.load_random_urls(fixed)
    print(f"Random URLs: {len(random_set)}")
    print(f"Total: {len(fixed) + len(random_set)}")
