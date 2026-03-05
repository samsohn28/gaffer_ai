"""
Fetch Understat xG/xA data for EPL players and save to data/raw/.

Understat exposes AJAX endpoints that return JSON directly:
  - getLeagueData/{league}/{year} -> season totals for all players
  - getPlayerData/{player_id}     -> full match history for one player

Requests must include the header X-Requested-With: XMLHttpRequest.

Two passes:
  1. League endpoint -> season totals for all EPL players
  2. Player endpoint -> per-match xG/xA for each player (filtered to current season)

Also builds an FPL <-> Understat ID map by fuzzy-matching player names.
"""

import json
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests

LEAGUE_DATA_URL = "https://understat.com/getLeagueData/EPL/{year}"
PLAYER_DATA_URL = "https://understat.com/getPlayerData/{player_id}"
AJAX_HEADERS = {"X-Requested-With": "XMLHttpRequest"}
CURRENT_SEASON_YEAR = "2024"  # 2024/25 season
REQUEST_DELAY = 0.5  # seconds between player requests

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "bronze"


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_league_players(year: str = CURRENT_SEASON_YEAR) -> list[dict]:
    url = LEAGUE_DATA_URL.format(year=year)
    resp = requests.get(url, headers=AJAX_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json().get("players", [])


def fetch_player_matches(player_id: str, year: str = CURRENT_SEASON_YEAR) -> list[dict]:
    url = PLAYER_DATA_URL.format(player_id=player_id)
    resp = requests.get(url, headers=AJAX_HEADERS, timeout=15)
    resp.raise_for_status()
    matches = resp.json().get("matches", [])
    current = [m for m in matches if m.get("season") == year]
    for m in current:
        m["player_id"] = player_id
    return current


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_json(data, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# FPL <-> Understat name matching
# ---------------------------------------------------------------------------

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def build_fpl_understat_map(
    fpl_elements: list[dict],
    understat_players: list[dict],
    threshold: float = 0.6,
) -> list[dict]:
    mapping = []
    for fpl in fpl_elements:
        fpl_full = f"{fpl['first_name']} {fpl['second_name']}"
        fpl_web = fpl["web_name"]

        best_score = 0.0
        best_us = None
        for us in understat_players:
            us_name = us["player_name"]
            score = max(
                _similarity(fpl_full, us_name),
                _similarity(fpl_web, us_name),
            )
            if score > best_score:
                best_score = score
                best_us = us

        if best_us is not None and best_score >= threshold:
            mapping.append({
                "fpl_id": fpl["id"],
                "understat_id": best_us["id"],
                "fpl_name": fpl_full,
                "understat_name": best_us["player_name"],
                "score": round(best_score, 4),
            })
        else:
            mapping.append({
                "fpl_id": fpl["id"],
                "understat_id": None,
                "fpl_name": fpl_full,
                "understat_name": best_us["player_name"] if best_us else None,
                "score": round(best_score, 4),
            })

    return mapping


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    year = CURRENT_SEASON_YEAR

    # 1. Season totals
    print(f"Fetching Understat EPL player data ({year}) ...")
    try:
        players = fetch_league_players(year)
    except requests.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    path = save_json(players, f"understat_players_{year}.json")
    print(f"  Saved {len(players)} players -> {path}")

    # 2. Per-match data
    print(f"Fetching per-match data for {len(players)} players ...")
    all_matches: list[dict] = []
    for i, player in enumerate(players, 1):
        pid = player["id"]
        try:
            matches = fetch_player_matches(pid, year)
            all_matches.extend(matches)
        except requests.RequestException as e:
            print(f"  Warning: failed for player {pid} ({player['player_name']}): {e}")
        if i % 50 == 0:
            print(f"  {i}/{len(players)} done ...")
        time.sleep(REQUEST_DELAY)

    path = save_json(all_matches, f"understat_matches_{year}.json")
    print(f"  Saved {len(all_matches)} match records -> {path}")

    # 3. FPL <-> Understat map
    print("Building FPL <-> Understat player map ...")
    elements_path = OUTPUT_DIR / "elements.json"
    if not elements_path.exists():
        print("  elements.json not found — run fpl_api.py first, skipping map.", file=sys.stderr)
    else:
        fpl_elements = json.loads(elements_path.read_text())
        mapping = build_fpl_understat_map(fpl_elements, players)
        matched = sum(1 for m in mapping if m["understat_id"] is not None)
        path = save_json(mapping, "understat_fpl_map.json")
        print(f"  Matched {matched}/{len(fpl_elements)} FPL players -> {path}")
        unmatched = [m for m in mapping if m["understat_id"] is None]
        if unmatched:
            print(f"  {len(unmatched)} unmatched (score < 0.6):")
            for m in unmatched[:10]:
                print(f"    fpl_id={m['fpl_id']} '{m['fpl_name']}' (best: '{m['understat_name']}', score={m['score']})")
            if len(unmatched) > 10:
                print(f"    ... and {len(unmatched) - 10} more")


if __name__ == "__main__":
    main()
