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
from urllib.parse import unquote

# Add src to path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.data.curated_articles import get_all_curated_articles


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

    def get_random_articles_batch(self, count: int = 10) -> List[str]:
        """
        Fetch multiple random Wikipedia article URLs in a single API call.

        Args:
            count: Number of articles to fetch (max 20 per API call).

        Returns:
            List of article URLs.
        """
        urls = []
        try:
            response = self.session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "list": "random",
                    "rnnamespace": 0,  # Main namespace only
                    "rnlimit": min(count, 20),  # API max is 20
                },
                timeout=15,
            )

            if response.status_code == 200:
                data = response.json()
                if "query" in data and "random" in data["query"]:
                    for item in data["query"]["random"]:
                        title = item["title"]
                        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        urls.append(url)
        except Exception as e:
            print(f"Batch API fetch error: {e}")

        return urls

    def get_random_article_via_api(self) -> str:
        """
        Fetch a random Wikipedia article URL using the MediaWiki API.

        Returns:
            Article URL or None if fetch failed.
        """
        batch = self.get_random_articles_batch(1)
        return batch[0] if batch else None

    def validate_article(self, url: str, verbose: bool = False) -> bool:
        """
        Pre-validate article by testing actual scraping capability.

        Args:
            url: Wikipedia article URL.
            verbose: Print validation details.

        Returns:
            True if article is valid (exists and has sufficient content).
        """
        try:
            # Properly decode URL-encoded characters
            raw_title = url.split("/wiki/")[-1]
            decoded_title = unquote(raw_title).replace("_", " ")

            page = self.wiki.page(decoded_title)

            if not page.exists():
                if verbose:
                    print(f"  ✗ Not found: {decoded_title[:50]}")
                return False

            # Check word count
            word_count = len(page.text.split())
            if word_count < 200:
                if verbose:
                    print(
                        f"  ✗ Too short ({word_count} words): {decoded_title[:50]}"
                    )
                return False

            if verbose:
                print(f"  ✓ Valid ({word_count} words): {decoded_title[:50]}")
            return True

        except Exception as e:
            if verbose:
                print(f"  ✗ Error: {e}")
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

        # Strategy 1: Featured/Good articles first (guaranteed quality)
        featured_target = min(count, 200)  # Quality articles are reliable
        pages.extend(
            self._fetch_featured_articles(featured_target, exclude_titles)
        )

        # Update exclude set (use lowercase for consistent dedup)
        for url in pages:
            raw_title = url.split("/wiki/")[-1]
            title = unquote(raw_title).replace("_", " ").lower()
            exclude_titles.add(title)

        # Strategy 2: Category-based for diversity (to reach target)
        if len(pages) < count:
            category_target = (
                count - len(pages) + 50
            )  # Fetch extra, some may be duplicates
            pages.extend(
                self._fetch_from_categories(category_target, exclude_titles)
            )

        print(f"\nSuccessfully fetched {len(pages)}/{count} valid articles.")
        return pages[:count]

    def _fetch_featured_articles(
        self, target: int, exclude_titles: Set[str]
    ) -> List[str]:
        """Fetch from Wikipedia's Featured and Good articles - guaranteed high quality."""
        pages = []
        gathered_titles = set()

        # Featured articles are guaranteed to be comprehensive
        quality_categories = [
            "Featured_articles",
            "Good_articles",
            "A-Class_articles",
        ]

        print(f"Fetching from quality article lists...")

        for cat_name in quality_categories:
            if len(pages) >= target:
                break

            try:
                cat = self.wiki.page(f"Category:{cat_name}")
                if not cat.exists():
                    continue

                for member in cat.categorymembers.values():
                    if len(pages) >= target:
                        break

                    # Skip subcategories, only get articles
                    if member.ns == wikipediaapi.Namespace.MAIN:
                        title_lower = member.title.lower()
                        if (
                            title_lower not in exclude_titles
                            and title_lower not in gathered_titles
                        ):
                            url = member.fullurl
                            pages.append(url)
                            gathered_titles.add(title_lower)
                            exclude_titles.add(
                                title_lower
                            )  # Prevent future duplicates
                            print(
                                f"Quality articles: {len(pages)}/{target}",
                                end="\r",
                            )

            except Exception as e:
                print(f"Error fetching {cat_name}: {e}")
                continue

        print(f"\nGot {len(pages)} from quality lists")
        return pages

    def _fetch_from_categories(
        self, target: int, exclude_titles: Set[str]
    ) -> List[str]:
        """Fetch articles from diverse Wikipedia categories."""
        # Expanded category list for better coverage - using more specific categories
        categories = [
            # Sciences
            "Physics",
            "Chemistry",
            "Biology",
            "Astronomy",
            "Mathematics",
            "Computer_science",
            "Medicine",
            "Geology",
            "Ecology",
            # History & Geography
            "History",
            "Geography",
            "Ancient_history",
            "World_War_II",
            # Arts & Culture
            "Art",
            "Literature",
            "Music",
            "Architecture",
            "Film",
            "Theatre",
            "Photography",
            "Sculpture",
            "Painting",
            # Social Sciences
            "Philosophy",
            "Psychology",
            "Economics",
            "Sociology",
            "Anthropology",
            "Linguistics",
            "Political_science",
            # Technology & Engineering
            "Technology",
            "Engineering",
            "Electronics",
            "Robotics",
            "Artificial_intelligence",
            "Software",
            "Internet",
            # Other
            "Sports",
            "Religion",
            "Mythology",
            "Cuisine",
            "Fashion",
            "Education",
            "Law",
            "Military",
            "Transportation",
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
                    if member.ns == wikipediaapi.Namespace.MAIN:
                        title_lower = member.title.lower()
                        if (
                            title_lower not in exclude_titles
                            and title_lower not in gathered_titles
                        ):
                            url = member.fullurl

                            # Pre-validate to avoid wasting slots
                            if self.validate_article(url):
                                pages.append(url)
                                gathered_titles.add(title_lower)
                                exclude_titles.add(
                                    title_lower
                                )  # Prevent future duplicates
                                print(
                                    f"Category: {len(pages)}/{target}",
                                    end="\r",
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
        self,
        target: int,
        exclude_titles: Set[str],
        existing_pages: List[str],
        skip_validation: bool = False,
    ) -> List[str]:
        """Fetch random articles using Wikipedia's Random API with batch fetching."""
        pages = []
        existing_urls = set(existing_pages)
        attempts = 0
        # Fetch extra to account for validation failures
        fetch_target = int(target * 1.5) if not skip_validation else target
        max_attempts = fetch_target * 10
        batch_size = 20

        print(
            f"\nFetching {target} articles via Random API (skip_validation={skip_validation})..."
        )

        while len(pages) < target and attempts < max_attempts:
            batch_urls = self.get_random_articles_batch(batch_size)

            for url in batch_urls:
                if len(pages) >= target:
                    break

                if url and url not in existing_urls:
                    title = url.split("/wiki/")[-1].replace("_", " ")
                    if title not in exclude_titles:
                        # Skip validation for speed - scraper will filter later
                        if skip_validation or self.validate_article(url):
                            pages.append(url)
                            existing_urls.add(url)
                            exclude_titles.add(title)
                            print(
                                f"Random API: {len(pages)}/{target}", end="\r"
                            )

            attempts += batch_size
            time.sleep(0.05)  # Faster rate limiting

        print(f"\nRandom API got {len(pages)} articles")
        return pages

    def fetch_curated_urls(self, target: int) -> List[str]:
        """
        Fetch URLs from curated article list for maximum diversity.

        Args:
            target: Number of URLs to fetch.

        Returns:
            List of validated Wikipedia URLs.
        """
        curated_titles = get_all_curated_articles()
        random.shuffle(curated_titles)  # Randomize order for variety

        valid_urls = []
        failed = []

        print(
            f"Fetching {target} URLs from {len(curated_titles)} curated articles..."
        )

        for title in curated_titles:
            if len(valid_urls) >= target:
                break

            try:
                page = self.wiki.page(title)
                if page.exists():
                    # Check content length
                    word_count = len(page.text.split())
                    if word_count >= 200:
                        valid_urls.append(page.fullurl)
                        print(
                            f"Curated: {len(valid_urls)}/{target} - {title[:40]}",
                            end="\r",
                        )
                    else:
                        failed.append(
                            f"{title} (too short: {word_count} words)"
                        )
                else:
                    failed.append(f"{title} (not found)")
            except Exception as e:
                failed.append(f"{title} (error: {e})")

        print(
            f"\nFetched {len(valid_urls)} curated URLs ({len(failed)} failed)"
        )
        return valid_urls

    def load_fixed_urls(self, force_refresh: bool = False) -> List[str]:
        """
        Load fixed URLs from JSON, or generate from curated list.
        Uses curated diverse articles for high quality.

        Args:
            force_refresh: If True, regenerate URLs even if file exists.

        Returns:
            List of fixed Wikipedia URLs.
        """
        # Load existing URLs if available and not forcing refresh
        if self.fixed_path.exists() and not force_refresh:
            print(f"Loading existing fixed URLs from {self.fixed_path}")
            with open(self.fixed_path, "r") as f:
                existing_urls = json.load(f)
            print(f"Found {len(existing_urls)} existing fixed URLs.")

            if len(existing_urls) >= self.fixed_count:
                return existing_urls[: self.fixed_count]

        # Generate fresh URLs from curated list
        print(
            f"Generating {self.fixed_count} diverse URLs from curated articles..."
        )
        urls = self.fetch_curated_urls(self.fixed_count)

        # Save
        with open(self.fixed_path, "w") as f:
            json.dump(urls, f, indent=2)
        print(f"Saved {len(urls)} fixed URLs to {self.fixed_path}")

        return urls[: self.fixed_count]

    def fetch_related_articles(
        self, source_urls: List[str], target: int
    ) -> List[str]:
        """
        Fetch related articles by extracting links from source pages.

        Args:
            source_urls: List of source Wikipedia URLs to get links from.
            target: Number of related articles to fetch.

        Returns:
            List of validated related Wikipedia URLs.
        """
        # Build exclude set from source URLs
        exclude_titles = set()
        for url in source_urls:
            try:
                raw_title = url.split("/wiki/")[-1]
                decoded_title = unquote(raw_title).replace("_", " ").lower()
                exclude_titles.add(decoded_title)
            except:
                pass

        # Collect all linked articles from source pages
        all_linked = []
        links_per_page = 5  # Get up to 5 links per source page

        print(
            f"Collecting related articles from {len(source_urls)} source pages..."
        )

        for i, url in enumerate(source_urls):
            try:
                raw_title = url.split("/wiki/")[-1]
                title = unquote(raw_title).replace("_", " ")
                page = self.wiki.page(title)

                if page.exists():
                    # Get links from this page
                    links = list(page.links.keys())[
                        :20
                    ]  # Sample from first 20 links
                    random.shuffle(links)

                    added = 0
                    for link_title in links:
                        if added >= links_per_page:
                            break

                        link_lower = link_title.lower()
                        # Skip if already in exclude set or is a meta page
                        if link_lower in exclude_titles:
                            continue
                        if ":" in link_title:  # Skip Wikipedia:, File:, etc.
                            continue

                        all_linked.append(link_title)
                        exclude_titles.add(link_lower)
                        added += 1

                print(
                    f"Collecting links: {i + 1}/{len(source_urls)} pages, {len(all_linked)} links",
                    end="\r",
                )

            except Exception as e:
                continue

        print(
            f"\nCollected {len(all_linked)} candidate links from source pages"
        )

        # Shuffle and validate to get target count
        random.shuffle(all_linked)

        valid_urls = []
        for title in all_linked:
            if len(valid_urls) >= target:
                break

            try:
                page = self.wiki.page(title)
                if page.exists():
                    word_count = len(page.text.split())
                    if word_count >= 200:
                        valid_urls.append(page.fullurl)
                        print(
                            f"Validating related: {len(valid_urls)}/{target}",
                            end="\r",
                        )
            except:
                continue

        print(f"\nValidated {len(valid_urls)} related articles")
        return valid_urls

    def load_random_urls(
        self, existing_urls: List[str], count: int = None
    ) -> List[str]:
        """
        Generate random URLs from articles related to the fixed set.
        Ensures thematic continuity while being random.

        Args:
            existing_urls: Fixed URLs to get related articles from.
            count: Optional override for number of URLs to fetch.

        Returns:
            List of random Wikipedia URLs (validated, no duplicates).
        """
        target = count if count is not None else self.random_count
        print(f"Fetching {target} random articles related to fixed set...")
        return self.fetch_related_articles(existing_urls, target)


if __name__ == "__main__":
    loader = URLLoader()
    fixed = loader.load_fixed_urls()
    print(f"\nFixed URLs: {len(fixed)}")

    random_set = loader.load_random_urls(fixed)
    print(f"Random URLs: {len(random_set)}")
    print(f"Total: {len(fixed) + len(random_set)}")
