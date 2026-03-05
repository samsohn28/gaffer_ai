"""
Fetch ClubElo team ratings for EPL clubs.

Endpoint: http://api.clubelo.com/{YYYY-MM-DD}
Returns a CSV of all European club ratings on that date.
No auth required, no rate limits.
"""

import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TEAMS_PATH = OUTPUT_DIR / "teams.json"
CLUBELO_URL = "http://api.clubelo.com/{date}"

# Map ClubElo club names to FPL team names
CLUBELO_TO_FPL = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "AstonVilla": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "CrystalPalace": "Crystal Palace",
    "Everton": "Everton",
    "Forest": "Nott'm Forest",
    "Fulham": "Fulham",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Man United": "Man Utd",
    "ManCity": "Man City",
    "ManUtd": "Man Utd",
    "Newcastle": "Newcastle",
    "NottmForest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Spurs": "Spurs",
    "Sunderland": "Sunderland",
    "Tottenham": "Spurs",
    "West Ham": "West Ham",
    "WestHam": "West Ham",
    "Wolves": "Wolves",
}


def fetch_ratings(date: str) -> list[dict]:
    """
    Fetch ClubElo ratings for the given date and filter to EPL (ENG, Level 1).

    Args:
        date: ISO date string, e.g. "2026-03-04"

    Returns:
        List of dicts with keys: club, elo
    """
    url = CLUBELO_URL.format(date=date)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    epl = []
    for row in reader:
        if row.get("Country") == "ENG" and row.get("Level") == "1":
            epl.append({"club": row["Club"], "elo": float(row["Elo"])})
    return epl


def map_to_fpl_names(ratings: list[dict]) -> dict[str, float]:
    """
    Map ClubElo club names to FPL team names.

    Unknown names are logged as warnings and skipped.

    Returns:
        Dict mapping FPL team name -> Elo rating
    """
    fpl_names = _load_fpl_team_names()
    result = {}
    for entry in ratings:
        club = entry["club"]
        fpl_name = CLUBELO_TO_FPL.get(club)
        if fpl_name is None:
            print(f"  Warning: no FPL mapping for ClubElo name '{club}'", file=sys.stderr)
            continue
        if fpl_names and fpl_name not in fpl_names:
            print(
                f"  Warning: '{fpl_name}' (mapped from '{club}') not found in teams.json",
                file=sys.stderr,
            )
        result[fpl_name] = round(entry["elo"], 1)
    return result


def _load_fpl_team_names() -> set[str]:
    """Load the set of FPL team names from teams.json, or empty set if unavailable."""
    if not TEAMS_PATH.exists():
        return set()
    teams = json.loads(TEAMS_PATH.read_text())
    return {t["name"] for t in teams}


def save_ratings(data: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "clubelo.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def main():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Fetching ClubElo ratings for {today}...")

    try:
        raw = fetch_ratings(today)
    except requests.RequestException as e:
        print(f"Error fetching ClubElo data: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(raw)} EPL clubs in ClubElo response")

    ratings = map_to_fpl_names(raw)

    if not ratings:
        print("Error: no ratings mapped — check CLUBELO_TO_FPL mapping.", file=sys.stderr)
        sys.exit(1)

    elo_values = list(ratings.values())
    output = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": today,
        "ratings": ratings,
    }

    path = save_ratings(output)
    print(
        f"Saved {len(ratings)} teams -> {path}"
        f"\n  Elo range: {min(elo_values):.1f} – {max(elo_values):.1f}"
    )
    top = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:3]
    print("  Top 3:", ", ".join(f"{name} ({elo})" for name, elo in top))


if __name__ == "__main__":
    main()
