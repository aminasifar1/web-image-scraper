#!/usr/bin/env python3
"""
Quick test of the corporate crawler with a small subset
"""
from pathlib import Path
from corporate_crawler import CorporateCrawler, setup_logging
import tempfile

# Create temp directory for test
with tempfile.TemporaryDirectory() as tmpdir:
    output_dir = Path(tmpdir)
    logger = setup_logging(output_dir)
    
    # Create a small test CSV
    test_csv = output_dir / "test_websites.csv"
    test_csv.write_text("""url,sector,subsector,organization_name
https://example.com,tech,web,Example
https://example.org,nonprofit,education,Example Org
""")
    
    # Create crawler
    crawler = CorporateCrawler(
        output_dir=output_dir,
        max_images_per_site=5,
        max_pages_per_site=2,
        delay_seconds=1.0,
        logger=logger,
    )
    
    # Load and crawl
    websites = crawler.load_websites(test_csv)
    stats = crawler.crawl_all(websites)
    
    print("\n✓ Test completed successfully!")
    print(f"  Images collected: {stats['images_downloaded']}")
    print(f"  Results in: {output_dir}")

if __name__ == "__main__":
    pass  # Run via: python -m pytest test_crawler.py -v
