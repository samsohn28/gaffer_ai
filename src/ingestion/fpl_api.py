"""
Fetch FPL bootstrap-static data and save to data/raw/.

The bootstrap-static endpoint returns a snapshot of the current season:
  - elements: all players and their stats
  - teams: all 20 PL clubs
  - events: gameweek schedule and deadlines
  - element_types: position definitions (GK, DEF, MID, FWD)
  - element_stats: stat definitions used for scoring
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import requests

API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"

KEYS_TO_SAVE = [
    "elements",
    "teams",
    "events",
    "element_types",
    "element_stats",
]


def fetch_bootstrap() -> dict:
    response = requests.get(API_URL, timeout=10)
    response.raise_for_status()
    return response.json()


def save(data: dict, key: str, timestamp: str) -> Path:
    path = OUTPUT_DIR / f"{key}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(data[key], f, indent=2)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    print(f"Fetching {API_URL} ...")
    try:
        data = fetch_bootstrap()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        sys.exit(1)

    for key in KEYS_TO_SAVE:
        if key not in data:
            print(f"  Warning: key '{key}' not found in response, skipping.")
            continue
        path = save(data, key, timestamp)
        print(f"  Saved {key} ({len(data[key]) if isinstance(data[key], list) else 1} records) -> {path}")


if __name__ == "__main__":
    main()
