"""
Scrape food images for YOLOv8 segmentation training.
Downloads images for each food class in the DiabeticAI prototype.
Output structure: dataset/<food_class>/img_001.jpg, img_002.jpg, ...

Usage:
    pip install icrawler
    python scrape_food_images.py

After downloading, upload each folder to Roboflow as a separate class.
"""

import os
import time
from icrawler.builtin import BingImageCrawler

# ── Configuration ──
IMAGES_PER_CLASS = 100          # number of images to download per food
OUTPUT_DIR = "dataset_round2"   # root output folder (separate from round 1)
MIN_SIZE = (200, 200)           # minimum image dimensions
MAX_NUM_THREADS = 4             # parallel download threads

# ── Food classes with NEW search queries (Round 2) ──
# Different queries from Round 1 to avoid duplicate images
FOOD_QUERIES = {
    "jollof_rice": [
        "Ghanaian jollof rice closeup",
        "party jollof rice tomato",
        "smoky jollof rice dish",
    ],
    "banku": [
        "banku okro stew Ghana",
        "banku ball fermented corn dough",
        "banku served with pepper",
    ],
    "fufu": [
        "cassava fufu bowl African",
        "fufu and light soup Ghana",
        "fufu plantain pounded dish",
    ],
    "waakye": [
        "waakye street food Accra",
        "waakye spaghetti shito egg",
        "waakye leaf wrapped rice",
    ],
    "kenkey": [
        "fante kenkey corn dough",
        "kenkey pepper fish Accra",
        "kenkey ball unwrapped plate",
    ],
    "plain_rice": [
        "cooked basmati rice dish",
        "white rice stew Africa",
        "jasmine rice plate served",
    ],
    "boiled_yam": [
        "yam slices boiled garden egg stew",
        "white yam boiled African food",
        "boiled yam kontomire Ghana",
    ],
    "grilled_tilapia": [
        "charcoal tilapia fish Accra",
        "roasted whole tilapia banku",
        "barbecue tilapia Ghana street food",
    ],
    "palm_nut_soup": [
        "banga soup palm fruit stew",
        "abenkwan palm soup Ghana",
        "palm cream soup rice fufu",
    ],
    "groundnut_soup": [
        "nkate nkwan peanut butter soup",
        "groundnut soup chicken Ghana homemade",
        "peanut stew African traditional",
    ],
    "okro_soup": [
        "okra stew slimy African food",
        "okro soup banku Ghana homemade",
        "fresh okra soup palm oil",
    ],
    "beans": [
        "red red beans Ghana fried plantain",
        "black eyed peas stew African",
        "cowpeas cooked tomato Ghana",
    ],
    "fried_plantain": [
        "kelewele spicy plantain cubes",
        "deep fried plantain golden African",
        "ripe plantain fried crispy Ghana",
    ],
    "grilled_chicken": [
        "suya chicken grilled spicy Africa",
        "chicken kebab charcoal Ghana",
        "roasted chicken drumstick African style",
    ],
    "fried_chicken": [
        "Ghana fried chicken Accra street",
        "deep fried chicken wings African",
        "crunchy fried chicken drumstick meal",
    ],
    "boiled_egg": [
        "sliced boiled egg on rice",
        "peeled hard boiled egg food",
        "boiled egg halves yolk plate",
    ],
    "shito": [
        "black chili oil Ghana condiment",
        "shito pepper fish oil dark sauce",
        "Ghanaian hot pepper shito homemade",
    ],
    "fried_fish": [
        "deep fried mackerel African plate",
        "crispy whole fried fish Ghana street",
        "fried herrings Ghana food",
    ],
    "red_pepper": [
        "scotch bonnet pepper Ghana",
        "African bird eye chili red",
        "hot red pepper sauce bowl",
    ],
    "plate": [
        "African food tray top view",
        "Ghana food plate variety overhead",
        "West African meal plate closeup",
    ],
}


def scrape_class(class_name, queries, images_per_class, output_dir):
    """Download images for a single food class."""
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    # Check how many images already exist (for resuming)
    existing = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    if existing >= images_per_class:
        print(f"  [{class_name}] Already has {existing} images, skipping.")
        return existing

    # Split target across queries
    per_query = max(images_per_class // len(queries), 10)

    total_downloaded = existing
    for i, query in enumerate(queries):
        if total_downloaded >= images_per_class:
            break

        remaining = images_per_class - total_downloaded
        count = min(per_query, remaining)

        print(f"  [{class_name}] Query {i+1}/{len(queries)}: \"{query}\" ({count} images)")

        crawler = BingImageCrawler(
            storage={"root_dir": class_dir},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=MAX_NUM_THREADS,
            log_level="WARNING",  # reduce noise
        )

        crawler.crawl(
            keyword=query,
            max_num=count,
            min_size=MIN_SIZE,
            file_idx_offset=total_downloaded,
        )

        # Count actual downloads
        current = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        total_downloaded = current

        # Brief pause between queries to avoid rate limiting
        time.sleep(1)

    return total_downloaded


def main():
    print("=" * 60)
    print("DiabeticAI - Food Image Scraper")
    print(f"Classes: {len(FOOD_QUERIES)}")
    print(f"Target: {IMAGES_PER_CLASS} images per class")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for i, (class_name, queries) in enumerate(FOOD_QUERIES.items(), 1):
        print(f"\n[{i}/{len(FOOD_QUERIES)}] Scraping: {class_name}")
        count = scrape_class(class_name, queries, IMAGES_PER_CLASS, OUTPUT_DIR)
        results[class_name] = count
        print(f"  -> {count} images in {class_name}/")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total = 0
    for cls, count in results.items():
        status = "OK" if count >= IMAGES_PER_CLASS * 0.7 else "LOW"
        print(f"  {cls:25s} {count:4d} images  [{status}]")
        total += count

    print(f"\n  Total: {total} images across {len(results)} classes")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}/")
    print("\nNext steps:")
    print("  1. Review images and remove irrelevant ones")
    print("  2. Upload each folder to Roboflow as a class")
    print("  3. Annotate with polygon masks for segmentation")


if __name__ == "__main__":
    main()
