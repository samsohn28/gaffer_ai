"""
Fetch FPL bootstrap-static and fixtures data and save to data/raw/.

The bootstrap-static endpoint returns a snapshot of the current season:
  - elements: all players and their stats
  - teams: all 20 PL clubs
  - events: gameweek schedule and deadlines
  - element_types: position definitions (GK, DEF, MID, FWD)
  - element_stats: stat definitions used for scoring

The fixtures endpoint returns all PL fixtures for the season:
  - event: gameweek number
  - team_h / team_a: home/away team IDs
  - team_h_difficulty / team_a_difficulty: FDR (1-5 scale)
  - finished / started: fixture status
  - team_h_score / team_a_score: result if played
"""

import json
import sys
from pathlib import Path

import requests

API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
LIVE_GW_URL = "https://fantasy.premierleague.com/api/event/{event_id}/live/"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "bronze"

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


def fetch_fixtures() -> list:
    response = requests.get(FIXTURES_URL, timeout=10)
    response.raise_for_status()
    return response.json()


def save(data: dict, key: str) -> Path:
    path = OUTPUT_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump(data[key], f, indent=2)
    return path


def current_event_id(events: list[dict]) -> int | None:
    """Return the ID of the current GW, or the most recently finished one."""
    for e in events:
        if e.get("is_current"):
            return e["id"]
    finished = [e for e in events if e.get("finished")]
    return finished[-1]["id"] if finished else None


def fetch_live_gw(event_id: int) -> dict:
    url = LIVE_GW_URL.format(event_id=event_id)
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def save_live_gw(data: dict, event_id: int) -> Path:
    path = OUTPUT_DIR / f"live_gw_{event_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def save_list(data: list, name: str) -> Path:
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        path = save(data, key)
        print(f"  Saved {key} ({len(data[key]) if isinstance(data[key], list) else 1} records) -> {path}")

    print(f"Fetching {FIXTURES_URL} ...")
    try:
        fixtures = fetch_fixtures()
    except requests.RequestException as e:
        print(f"Error fetching fixtures: {e}", file=sys.stderr)
        sys.exit(1)

    path = save_list(fixtures, "fixtures")
    print(f"  Saved fixtures ({len(fixtures)} records) -> {path}")

    events = json.loads((OUTPUT_DIR / "events.json").read_text())
    event_id = current_event_id(events)
    if event_id is None:
        print("  No current or finished gameweek found, skipping live data.")
    else:
        print(f"Fetching live GW data for event {event_id} ...")
        try:
            live = fetch_live_gw(event_id)
        except requests.RequestException as e:
            print(f"Error fetching live GW data: {e}", file=sys.stderr)
        else:
            path = save_live_gw(live, event_id)
            print(f"  Saved live_gw_{event_id} ({len(live.get('elements', []))} elements) -> {path}")


if __name__ == "__main__":
    main()
