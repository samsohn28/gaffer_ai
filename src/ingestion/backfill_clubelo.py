"""
Backfill ClubElo ratings for each historical GW.

For each finished GW, fetches ClubElo ratings on the date of the first kickoff
and saves them to data/bronze/clubelo_historical.json.

Run:
    python -m src.ingestion.backfill_clubelo
"""

import json
import sys
from pathlib import Path

import pandas as pd
import requests

from src.ingestion.clubelo import fetch_ratings, map_to_fpl_names

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
OUTPUT_PATH = BRONZE / "clubelo_historical.json"


def load_existing() -> dict[int, dict]:
    """Load already-fetched GW entries keyed by GW number."""
    if not OUTPUT_PATH.exists():
        return {}
    data = json.loads(OUTPUT_PATH.read_text())
    return {entry["gw"]: entry for entry in data}


def main():
    fixtures = pd.read_parquet(ROOT / "data" / "silver" / "fixtures.parquet")

    # Finished GWs with a valid kickoff date
    finished = fixtures[fixtures["finished"] == True].copy()
    finished = finished.dropna(subset=["event", "kickoff_date"])
    finished["event"] = finished["event"].astype(int)

    # First kickoff date per GW
    gw_dates = (
        finished.groupby("event")["kickoff_date"]
        .min()
        .reset_index()
        .rename(columns={"event": "gw", "kickoff_date": "date"})
        .sort_values("gw")
    )

    existing = load_existing()
    results = dict(existing)

    for _, row in gw_dates.iterrows():
        gw = int(row["gw"])
        date = str(row["date"])

        if gw in existing:
            print(f"GW{gw:2d} ({date}): already fetched, skipping")
            continue

        print(f"GW{gw:2d} ({date}): fetching...", end=" ", flush=True)
        try:
            raw = fetch_ratings(date)
            ratings = map_to_fpl_names(raw)
            results[gw] = {"gw": gw, "date": date, "ratings": ratings}
            print(f"{len(ratings)} teams")
        except requests.RequestException as e:
            print(f"ERROR - {e}", file=sys.stderr)

    # Save sorted by GW
    ordered = sorted(results.values(), key=lambda x: x["gw"])
    BRONZE.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(ordered, indent=2))
    print(f"\nSaved {len(ordered)} GW entries -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
