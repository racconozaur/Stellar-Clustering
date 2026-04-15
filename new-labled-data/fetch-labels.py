import csv
import json
import logging
import os
import time
import requests

BASE_URL = "https://api.stellar.expert"
START_PATH = "/explorer/public/directory?limit=200&order=asc"

JSON_FILE = "stellar_directory.json"
CSV_FILE = "stellar_directory.csv"
CHECKPOINT_FILE = "stellar_directory.checkpoint.json"
LOG_FILE = "fetch-labels.log"

HEADERS = {"accept": "application/json"}

MAX_RETRIES = 8            # was 5
BASE_BACKOFF = 2.0         # seconds; exponential: 2,4,8,16,32,64,128,256
PER_PAGE_DELAY = 1.0       # polite gap between successful pages (was 0.2)
REQUEST_TIMEOUT = 30


# ---------- logging ----------
logger = logging.getLogger("stellar_directory")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

_fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_sh)


# ---------- checkpointing ----------
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return {"next_url": BASE_URL + START_PATH, "records": []}
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(state):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    os.replace(tmp, CHECKPOINT_FILE)


# ---------- fetching ----------
def get_with_retries(url):
    """GET with exponential backoff. Honors Retry-After header if present."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            wait = BASE_BACKOFF * (2 ** attempt)
            logger.warning(f"  network error {e}; sleeping {wait:.0f}s")
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            return resp

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = BASE_BACKOFF * (2 ** attempt)
            logger.warning(f"  rate limited (429); sleeping {wait:.0f}s "
                           f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue

        # other HTTP errors: brief retry, then raise
        logger.error(f"  HTTP {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()

    raise RuntimeError(f"Exceeded {MAX_RETRIES} retries for {url}")


def fetch_full_directory():
    state = load_checkpoint()
    url = state["next_url"]
    all_records = state["records"]
    page = 0

    if all_records:
        logger.info(f"Resuming from checkpoint: {len(all_records)} records already fetched")

    while url:
        page += 1
        logger.info(f"[page {page}] GET {url}")

        resp = get_with_retries(url)
        data = resp.json()
        records = data.get("_embedded", {}).get("records", [])

        if not records:
            logger.info("  empty page; stopping")
            break

        all_records.extend(records)
        logger.info(f"  got {len(records)} records (total: {len(all_records)})")

        next_href = data.get("_links", {}).get("next", {}).get("href")
        url = BASE_URL + next_href if next_href else None

        # persist progress after every page
        save_checkpoint({"next_url": url, "records": all_records})

        if url:
            time.sleep(PER_PAGE_DELAY)

    logger.info(f"Total directory entries fetched: {len(all_records)}")
    return all_records


# ---------- output ----------
def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON -> {filename}")


def save_to_csv(data, filename):
    fieldnames = ["address", "name", "domain", "tags", "paging_token"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow({
                "address": entry.get("address", ""),
                "name": entry.get("name", ""),
                "domain": entry.get("domain", ""),
                "tags": ", ".join(entry.get("tags", []) or []),
                "paging_token": entry.get("paging_token", ""),
            })
    logger.info(f"Saved CSV  -> {filename}")


def summarize(data):
    tag_counts = {}
    for entry in data:
        for t in entry.get("tags", []) or []:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    logger.info("Tag distribution:")
    for tag, n in sorted(tag_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {tag:30s} {n}")


def main():
    logger.info("=" * 60)
    logger.info("Starting StellarExpert directory fetch")
    try:
        directory = fetch_full_directory()
    except Exception as e:
        logger.exception(f"Fetch failed: {e}")
        logger.info("Checkpoint preserved; re-run the script to resume.")
        raise

    save_to_json(directory, JSON_FILE)
    save_to_csv(directory, CSV_FILE)
    summarize(directory)

    # success: remove checkpoint so next run starts fresh
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logger.info("Checkpoint cleared.")


if __name__ == "__main__":
    main()