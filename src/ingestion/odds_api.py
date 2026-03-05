"""
Fetch betting market odds from The Odds API for upcoming EPL fixtures.

Endpoints used:
  GET /v4/sports/soccer_epl/events                          — list upcoming fixtures
  GET /v4/sports/soccer_epl/odds?markets=h2h,btts           — match result + BTTS odds
  GET /v4/sports/soccer_epl/events/{id}/odds?markets=...    — anytime goalscorer props

Requires ODDS_API_KEY in environment (or .env file).
Total requests per GW run: ~12, well within 500/month free tier.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "soccer_epl"
API_KEY = os.environ.get("ODDS_API_KEY")
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "bronze"
MAX_RETRIES = 3


def _get(url: str, params: dict, max_retries: int = MAX_RETRIES) -> requests.Response:
    """GET with automatic retry on 429, honouring Retry-After if present."""
    for attempt in range(max_retries):
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("Retry-After")
        wait = int(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
        print(f"  Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
        time.sleep(wait)
    resp.raise_for_status()
    return resp


def fetch_events() -> tuple[list[dict], dict]:
    """Return upcoming EPL fixtures and response headers (for quota tracking)."""
    url = f"{BASE_URL}/sports/{SPORT}/events"
    resp = _get(url, {"apiKey": API_KEY})
    return resp.json(), resp.headers


def fetch_match_odds(event_ids: list[str]) -> tuple[list[dict], dict]:
    """Fetch h2h odds for all given event IDs in one request."""
    url = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "markets": "h2h",
        "eventIds": ",".join(event_ids),
        "oddsFormat": "decimal",
        "regions": "uk",
    }
    resp = _get(url, params)
    return resp.json(), resp.headers


def fetch_event_odds(event_id: str) -> tuple[dict, dict]:
    """Fetch btts + anytime goalscorer props for one fixture."""
    url = f"{BASE_URL}/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "markets": "btts,player_goal_scorer_anytime",
        "oddsFormat": "decimal",
        "regions": "eu",
    }
    resp = _get(url, params)
    return resp.json(), resp.headers


def to_implied_prob(outcomes: list[dict]) -> dict[str, float]:
    """Convert decimal odds to de-vigged implied probabilities."""
    raw = {o["name"]: 1 / o["price"] for o in outcomes}
    total = sum(raw.values())
    return {name: p / total for name, p in raw.items()}


def build_clean_sheet_probs(
    event_id: str,
    h2h_data: list[dict],
    event_odds: dict,
) -> dict[str, float]:
    """
    Derive per-team clean sheet probability from h2h + both_teams_to_score markets.

    Approximation:
      P(home CS) ≈ P(home win or draw) × P(BTTS=No)
      P(away CS) ≈ P(away win or draw) × P(BTTS=No)

    h2h_data: batch response from fetch_match_odds (list of events)
    event_odds: per-event response from fetch_event_odds (single event dict)
    """
    h2h_event = next((e for e in h2h_data if e["id"] == event_id), None)
    home_team = event_odds.get("home_team", "")
    away_team = event_odds.get("away_team", "")

    # Extract h2h outcomes from batch data
    h2h_outcomes = None
    if h2h_event:
        for bookmaker in h2h_event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] == "h2h":
                    h2h_outcomes = market["outcomes"]
                    break
            if h2h_outcomes:
                break

    # Extract BTTS outcomes from per-event data
    btts_outcomes = None
    for bookmaker in event_odds.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market["key"] == "btts":
                btts_outcomes = market["outcomes"]
                break
        if btts_outcomes:
            break

    if not h2h_outcomes or not btts_outcomes:
        return {"home": 0.0, "away": 0.0}

    h2h_probs = to_implied_prob(h2h_outcomes)
    btts_probs = to_implied_prob(btts_outcomes)
    p_btts_no = btts_probs.get("No", 0.0)

    p_home_no_concede = h2h_probs.get(home_team, 0.0) + h2h_probs.get("Draw", 0.0)
    p_away_no_concede = h2h_probs.get(away_team, 0.0) + h2h_probs.get("Draw", 0.0)

    return {
        "home": round(p_home_no_concede * p_btts_no, 4),
        "away": round(p_away_no_concede * p_btts_no, 4),
    }


def build_goalscorer_probs(event_data: dict) -> list[dict]:
    """
    Parse anytime goalscorer market and return de-vigged probabilities per player.
    Note: probs do NOT sum to 1 — multiple players can score in the same game.
    """
    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")

    # Aggregate raw probabilities across bookmakers, then average
    player_probs: dict[str, list[float]] = {}
    for bookmaker in event_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "player_goal_scorer_anytime":
                continue
            outcomes = [o for o in market["outcomes"] if o["name"] not in ("Yes", "No")]
            if not outcomes:
                continue
            devigged = to_implied_prob(outcomes)
            for name, prob in devigged.items():
                player_probs.setdefault(name, []).append(prob)

    if not player_probs:
        return []

    result = []
    for player_name, probs in player_probs.items():
        avg_prob = sum(probs) / len(probs)
        # Heuristic: assign team based on which team name appears in description
        # The Odds API includes a "description" field with team name in some responses
        team = _guess_team(player_name, event_data.get("bookmakers", []), home_team, away_team)
        result.append({
            "player_name": player_name,
            "team": team,
            "prob": round(avg_prob, 4),
        })

    return sorted(result, key=lambda x: x["prob"], reverse=True)


def _guess_team(player_name: str, bookmakers: list[dict], home_team: str, away_team: str) -> str:
    """Extract team from outcome description field if available, else return empty string."""
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market["key"] != "player_goal_scorer_anytime":
                continue
            for outcome in market.get("outcomes", []):
                if outcome.get("name") == player_name:
                    desc = outcome.get("description", "")
                    if home_team and home_team in desc:
                        return home_team
                    if away_team and away_team in desc:
                        return away_team
                    return desc  # use raw description as fallback
    return ""


def next_gameweek(events_path: Path) -> int:
    """Return the next upcoming GW number from events.json."""
    events = json.loads(events_path.read_text())
    for e in events:
        if e.get("is_next"):
            return e["id"]
    # Fallback: first unfinished event
    for e in events:
        if not e.get("finished"):
            return e["id"]
    # All finished — return last id + 1
    return events[-1]["id"] + 1


def save_odds(data: dict, gw: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"odds_gw_{gw}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def requests_remaining(headers: dict) -> int | None:
    val = headers.get("x-requests-remaining")
    return int(val) if val is not None else None


def main():
    if not API_KEY:
        print("Error: ODDS_API_KEY not set. Add it to .env or your environment.", file=sys.stderr)
        sys.exit(1)

    events_path = OUTPUT_DIR / "events.json"
    if not events_path.exists():
        print("Error: data/raw/events.json not found. Run fpl_api.py first.", file=sys.stderr)
        sys.exit(1)

    gw = next_gameweek(events_path)
    print(f"Target gameweek: GW{gw}")

    # 1. Fetch upcoming fixtures
    print("Fetching upcoming EPL fixtures...")
    try:
        events, hdrs = fetch_events()
    except requests.RequestException as e:
        print(f"Error fetching events: {e}", file=sys.stderr)
        sys.exit(1)

    if not events:
        print("No upcoming fixtures found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Found {len(events)} upcoming fixture(s)")
    event_ids = [e["id"] for e in events]

    # 2. Fetch match odds (h2h + btts) in one request
    print("Fetching match odds (h2h + btts)...")
    try:
        match_odds_data, hdrs = fetch_match_odds(event_ids)
    except requests.RequestException as e:
        print(f"Error fetching match odds: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Fetch goalscorer props per fixture
    fixtures_out = []
    total_goalscorer_players = 0

    for event in events:
        eid = event["id"]
        home = event["home_team"]
        away = event["away_team"]
        print(f"  Fetching event odds: {home} vs {away}...")

        try:
            event_odds, hdrs = fetch_event_odds(eid)
        except requests.RequestException as e:
            print(f"    Warning: failed to fetch event odds for {eid}: {e}")
            event_odds = {}

        cs_probs = build_clean_sheet_probs(eid, match_odds_data, event_odds)
        goalscorer_probs = build_goalscorer_probs(event_odds)
        total_goalscorer_players += len(goalscorer_probs)

        fixtures_out.append({
            "event_id": eid,
            "home_team": home,
            "away_team": away,
            "commence_time": event.get("commence_time"),
            "clean_sheet_prob": cs_probs,
            "goalscorer_probs": goalscorer_probs,
        })

    remaining = requests_remaining(hdrs)
    output = {
        "gameweek": gw,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "requests_remaining": remaining,
        "fixtures": fixtures_out,
    }

    path = save_odds(output, gw)
    print(f"\nSaved {len(fixtures_out)} fixture(s), {total_goalscorer_players} total goalscorer entries -> {path}")
    if remaining is not None:
        print(f"Requests remaining this month: {remaining}")


if __name__ == "__main__":
    main()
