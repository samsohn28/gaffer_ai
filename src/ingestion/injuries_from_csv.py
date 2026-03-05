"""
Parse a premierinjuries.com CSV export and save to data/raw/injuries.json.

Expects the most recent CSV in data/manual/ matching premierinjuries_*.csv.
Columns: Player, Reason, Further Detail, Potential Return, Condition, Status, Team

Usage:
    python -m src.ingestion.injuries_from_csv
"""

import csv
import difflib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MANUAL_DIR = Path(__file__).resolve().parents[2] / "data" / "manual"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_PATH = RAW_DIR / "injuries.json"
ELEMENTS_PATH = RAW_DIR / "elements.json"

# Maps CSV team slug -> FPL team name (as used in teams.json "name" field)
TEAM_SLUG_TO_FPL = {
    "arsenal": "Arsenal",
    "astonvilla": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton",
    "burnley": "Burnley",
    "chelsea": "Chelsea",
    "crystalpalace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "leeds": "Leeds",
    "liverpool": "Liverpool",
    "manchestercity": "Man City",
    "mancity": "Man City",
    "manchesterunited": "Man Utd",
    "manunited": "Man Utd",
    "newcastle": "Newcastle",
    "nottinghamforest": "Nott'm Forest",
    "sunderland": "Sunderland",
    "tottenham": "Spurs",
    "westham": "West Ham",
    "wolves": "Wolves",
}


def parse_csv(path: Path) -> list[dict]:
    """Read the injury CSV and return a list of normalised dicts."""
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status_raw = row["Status"].strip()
            if status_raw.lower() == "ruled out":
                confidence = 0
            else:
                digits = "".join(c for c in status_raw if c.isdigit())
                confidence = min(int(digits), 100) if digits else 0

            team_slug = row.get("Team", "").strip().lower().replace(" ", "")
            team = TEAM_SLUG_TO_FPL.get(team_slug, "")
            if team_slug and not team:
                print(f"  Warning: no FPL mapping for team slug '{team_slug}'", file=sys.stderr)

            records.append({
                "player_name": row["Player"].strip(),
                "team": team,
                "reason": row["Reason"].strip(),
                "further_detail": row["Further Detail"].strip(),
                "potential_return": row["Potential Return"].strip(),
                "condition": row["Condition"].strip(),
                "confidence": confidence,
            })
    return records


def map_to_fpl_ids(injuries: list[dict]) -> list[dict]:
    """Add fpl_id to each record by matching web_name (team-scoped) in elements.json."""
    if not ELEMENTS_PATH.exists():
        print(
            f"  Warning: {ELEMENTS_PATH} not found — skipping FPL ID mapping.",
            file=sys.stderr,
        )
        for r in injuries:
            r["fpl_id"] = None
        return injuries

    elements = json.loads(ELEMENTS_PATH.read_text())

    # Build team name -> set of FPL team ids
    teams_path = RAW_DIR / "teams.json"
    fpl_name_to_id: dict[str, int] = {}
    if teams_path.exists():
        teams = json.loads(teams_path.read_text())
        fpl_name_to_id = {t["name"]: t["id"] for t in teams}

    # Index elements by web_name and full name; also build per-team index
    name_to_players: dict[str, list[dict]] = {}
    team_id_to_elements: dict[int, list[dict]] = {}
    for el in elements:
        name_to_players.setdefault(el["web_name"], []).append(el)
        full_name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
        if full_name and full_name != el["web_name"]:
            name_to_players.setdefault(full_name, []).append(el)
        team_id_to_elements.setdefault(el["team"], []).append(el)

    all_web_names = list(name_to_players.keys())

    for record in injuries:
        player_name = record["player_name"]
        fpl_team_name = record.get("team", "")
        fpl_team_id = fpl_name_to_id.get(fpl_team_name) if fpl_team_name else None

        # Candidates restricted to correct team when we have team info
        team_elements = team_id_to_elements.get(fpl_team_id, []) if fpl_team_id else elements
        team_name_map: dict[str, dict] = {}
        for el in team_elements:
            team_name_map[el["web_name"]] = el
            full = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
            if full:
                team_name_map[full] = el
        team_web_names = list(team_name_map.keys())

        matched_id = None

        # 1. Exact match within team
        if player_name in team_name_map:
            matched_id = team_name_map[player_name]["id"]
        # 2. Fuzzy match within team
        elif team_web_names:
            close = difflib.get_close_matches(player_name, team_web_names, n=1, cutoff=0.5)
            if close:
                matched_id = team_name_map[close[0]]["id"]
                print(f"  Fuzzy match: '{player_name}' -> '{close[0]}' (id={matched_id})")
        # 3. Fallback: fuzzy across all players (no team info or team not found)
        if matched_id is None and not fpl_team_id:
            close = difflib.get_close_matches(player_name, all_web_names, n=1, cutoff=0.6)
            if close:
                matched_id = name_to_players[close[0]][0]["id"]
                print(f"  Fuzzy match (no team): '{player_name}' -> '{close[0]}' (id={matched_id})")

        if matched_id is None:
            print(f"  Warning: no FPL match for '{player_name}' ({fpl_team_name})", file=sys.stderr)

        record["fpl_id"] = matched_id

    return injuries


def save_injuries(injuries: list[dict], source_file: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "manual_csv",
        "source_file": source_file,
        "injuries": injuries,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    return OUTPUT_PATH


def main():
    csvs = sorted(MANUAL_DIR.glob("premierinjuries_*.csv"))
    if not csvs:
        print(f"No premierinjuries_*.csv found in {MANUAL_DIR}", file=sys.stderr)
        sys.exit(1)

    csv_path = csvs[-1]  # most recent by filename sort
    print(f"Parsing {csv_path.name}...")

    injuries = parse_csv(csv_path)
    print(f"  {len(injuries)} players read from CSV")

    injuries = map_to_fpl_ids(injuries)

    unmatched = sum(1 for r in injuries if r["fpl_id"] is None)
    path = save_injuries(injuries, csv_path.name)

    print(f"\nSaved -> {path}")
    print(f"  Players total : {len(injuries)}")
    print(f"  FPL ID misses : {unmatched}")
    match_rate = (len(injuries) - unmatched) / len(injuries) * 100 if injuries else 0
    print(f"  Match rate    : {match_rate:.1f}%")


if __name__ == "__main__":
    main()
