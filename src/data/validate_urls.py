"""
URL Validation and Cleanup Script.

Validates existing fixed_urls.json, removes invalid URLs,
and fetches replacements to maintain target count.
"""

import json
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import Config
from src.data.url_loader import URLLoader


def validate_and_clean_fixed_urls():
    """Validate existing fixed URLs and replace invalid ones."""
    loader = URLLoader()
    fixed_path = Config.FIXED_URLS_PATH
    target_count = 200
    
    # Load existing URLs
    if not fixed_path.exists():
        print("No fixed_urls.json found. Run url_loader first.")
        return
    
    with open(fixed_path, "r") as f:
        urls = json.load(f)
    
    print(f"Loaded {len(urls)} URLs from fixed_urls.json")
    print("Validating URLs (this tests actual scraping capability)...\n")
    
    valid_urls = []
    invalid_urls = []
    seen_titles = set()
    
    for url in tqdm(urls, desc="Validating"):
        # Extract title for dedup check
        raw_title = url.split("/wiki/")[-1]
        from urllib.parse import unquote
        title = unquote(raw_title).replace("_", " ").lower()
        
        # Skip duplicates
        if title in seen_titles:
            print(f"  ✗ Duplicate: {title[:50]}")
            invalid_urls.append(url)
            continue
        
        # Validate
        if loader.validate_article(url, verbose=False):
            valid_urls.append(url)
            seen_titles.add(title)
        else:
            invalid_urls.append(url)
            print(f"  ✗ Invalid: {title[:60]}")
    
    print(f"\n--- Validation Results ---")
    print(f"Valid URLs: {len(valid_urls)}")
    print(f"Invalid URLs: {len(invalid_urls)}")
    
    # Need to fetch replacements?
    needed = target_count - len(valid_urls)
    if needed > 0:
        print(f"\nFetching {needed} replacement URLs...")
        
        # Build exclude set from valid URLs
        exclude_titles = set()
        for url in valid_urls:
            raw_title = url.split("/wiki/")[-1]
            title = unquote(raw_title).replace("_", " ")
            exclude_titles.add(title)
        
        # Fetch replacements with validation
        new_urls = fetch_validated_urls(loader, needed, exclude_titles)
        valid_urls.extend(new_urls)
        print(f"Added {len(new_urls)} new URLs")
    
    # Save cleaned URLs
    print(f"\nSaving {len(valid_urls)} validated URLs...")
    with open(fixed_path, "w") as f:
        json.dump(valid_urls[:target_count], f, indent=2)
    
    print(f"✓ Saved {min(len(valid_urls), target_count)} URLs to {fixed_path}")
    return valid_urls[:target_count]


def fetch_validated_urls(loader, count, exclude_titles):
    """Fetch URLs and validate each one before accepting."""
    valid_urls = []
    attempts = 0
    max_attempts = count * 5
    
    # Use quality articles as primary source
    print("Fetching from quality article categories...")
    quality_cats = ["Featured_articles", "Good_articles"]
    
    for cat_name in quality_cats:
        if len(valid_urls) >= count:
            break
            
        try:
            cat = loader.wiki.page(f"Category:{cat_name}")
            if not cat.exists():
                continue
            
            import wikipediaapi
            for member in cat.categorymembers.values():
                if len(valid_urls) >= count:
                    break
                
                if member.ns == wikipediaapi.Namespace.MAIN:
                    title = member.title
                    if title.lower() not in {t.lower() for t in exclude_titles}:
                        url = member.fullurl
                        if loader.validate_article(url):
                            valid_urls.append(url)
                            exclude_titles.add(title)
                            print(f"  ✓ Added: {title[:50]}")
                            
        except Exception as e:
            print(f"Error with {cat_name}: {e}")
    
    return valid_urls


if __name__ == "__main__":
    validate_and_clean_fixed_urls()
