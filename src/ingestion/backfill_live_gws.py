"""
Backfill historical live GW data from the FPL API.

For every finished gameweek that doesn't already have a live_gw_{n}.json in
data/bronze/, fetch it from the FPL API and save it.

Run:
    python -m src.ingestion.backfill_live_gws
"""

import json
import sys
import time
from pathlib import Path

import requests

LIVE_GW_URL = "https://fantasy.premierleague.com/api/event/{event_id}/live/"
BRONZE = Path(__file__).resolve().parents[2] / "data" / "bronze"


def finished_gw_ids(events: list[dict]) -> list[int]:
    return [e["id"] for e in events if e.get("finished")]


def main():
    events_path = BRONZE / "events.json"
    if not events_path.exists():
        print("events.json not found — run fpl_api.py first.", file=sys.stderr)
        sys.exit(1)

    events = json.loads(events_path.read_text())
    gw_ids = finished_gw_ids(events)
    print(f"Finished GWs: {gw_ids}")

    to_fetch = [gw for gw in gw_ids if not (BRONZE / f"live_gw_{gw}.json").exists()]
    if not to_fetch:
        print("All finished GW files already present — nothing to fetch.")
        return

    print(f"Fetching {len(to_fetch)} missing GW(s): {to_fetch}")
    for gw in to_fetch:
        url = LIVE_GW_URL.format(event_id=gw)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  GW{gw}: ERROR — {e}", file=sys.stderr)
            continue

        out = BRONZE / f"live_gw_{gw}.json"
        out.write_text(json.dumps(data))
        n = len(data.get("elements", []))
        print(f"  GW{gw}: {n} elements → {out.name}")

        # Be polite to the FPL API
        time.sleep(1.0)

    print("Done.")


if __name__ == "__main__":
    main()
