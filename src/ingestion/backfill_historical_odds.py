"""
Fetch historical h2h odds for GW1-29 from The Odds API historical endpoint.

Snapshot time: 08:00 UTC on first match day of each GW (pre-match).
Cost: 10 credits per GW × 29 GWs = 290 credits.

Saves to data/bronze/odds_historical.json.

Run:
    python -m src.ingestion.backfill_historical_odds
"""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"
API_KEY = os.environ.get("ODDS_API_KEY")
BRONZE = Path(__file__).resolve().parents[2] / "data" / "bronze"
SILVER = Path(__file__).resolve().parents[2] / "data" / "silver"

# Odds API team name → FPL team name
ODDS_TO_FPL = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Tottenham Hotspur": "Spurs",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    # Direct matches
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Liverpool": "Liverpool",
    "Sunderland": "Sunderland",
    "Southampton": "Southampton",
}


def normalize(name: str) -> str:
    return ODDS_TO_FPL.get(name, name)


def to_implied_prob(outcomes: list[dict]) -> dict[str, float]:
    raw = {o["name"]: 1 / o["price"] for o in outcomes}
    total = sum(raw.values())
    return {name: round(p / total, 4) for name, p in raw.items()}


def fetch_historical_h2h(date_str: str) -> tuple[list[dict], dict]:
    """Fetch h2h odds snapshot at the given ISO8601 date string."""
    url = f"{BASE_URL}/historical/sports/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "markets": "h2h",
        "regions": "uk",
        "date": date_str,
        "oddsFormat": "decimal",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json(), resp.headers


def gw_first_dates(fixtures_path: Path) -> dict[int, str]:
    """Return {gw: 'YYYY-MM-DDT08:00:00Z'} for finished GWs."""
    fix = pd.read_parquet(fixtures_path)
    finished = fix[fix["finished"] == True].dropna(subset=["event", "kickoff_date"])
    result = {}
    for gw, grp in finished.groupby("event"):
        first_date = grp["kickoff_date"].min()
        result[int(gw)] = f"{first_date}T08:00:00Z"
    return result


def parse_fixtures(events: list[dict]) -> list[dict]:
    """Extract win probs per fixture from h2h response."""
    rows = []
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        h2h_outcomes = None
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    h2h_outcomes = market["outcomes"]
                    break
            if h2h_outcomes:
                break
        if not h2h_outcomes:
            continue
        probs = to_implied_prob(h2h_outcomes)
        rows.append({
            "home_team": home,
            "away_team": away,
            "home_fpl": normalize(home),
            "away_fpl": normalize(away),
            "home_win_prob": probs.get(home, 0.0),
            "draw_prob": probs.get("Draw", 0.0),
            "away_win_prob": probs.get(away, 0.0),
        })
    return rows


def main():
    if not API_KEY:
        print("Error: ODDS_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    fixtures_path = SILVER / "fixtures.parquet"
    if not fixtures_path.exists():
        print("Error: fixtures.parquet not found. Run silver pipeline first.", file=sys.stderr)
        sys.exit(1)

    gw_dates = gw_first_dates(fixtures_path)
    gws = sorted(gw_dates.keys())
    print(f"Fetching historical h2h odds for {len(gws)} GWs...")

    all_records = []
    last_headers = {}

    for gw in gws:
        date_str = gw_dates[gw]
        print(f"  GW{gw:2d}  {date_str[:10]} ...", end=" ", flush=True)
        try:
            data, last_headers = fetch_historical_h2h(date_str)
        except requests.RequestException as e:
            print(f"ERROR: {e}")
            continue

        events = data.get("data", [])
        fixtures = parse_fixtures(events)
        for f in fixtures:
            f["gw"] = gw
        all_records.extend(fixtures)
        remaining = last_headers.get("x-requests-remaining", "?")
        print(f"{len(fixtures)} fixtures  (remaining: {remaining})")
        time.sleep(0.3)

    out = BRONZE / "odds_historical.json"
    out.write_text(json.dumps(all_records, indent=2))
    print(f"\nSaved {len(all_records)} fixture records → {out}")
    remaining = last_headers.get("x-requests-remaining")
    if remaining:
        print(f"Requests remaining: {remaining}")


if __name__ == "__main__":
    main()
