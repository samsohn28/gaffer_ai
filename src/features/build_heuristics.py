"""
Build heuristic player predictions from raw FPL bootstrap-static data.

Expected points formula:
    weighted_score = (0.6 * points_per_game) + (0.4 * form)
    expected_points = weighted_score * chance_of_playing

Notes:
  - `points_per_game` is the season-long average from the FPL API (string).
  - `form` is the FPL-provided 4-gameweek rolling average (string).
  - `chance_of_playing_next_round` is 0-100 or None (None means fully available → 1.0).
  - `now_cost` is stored in tenths by FPL (e.g. 65 → £6.5m).
  - Players with 0 minutes are excluded (no meaningful PPG).
"""

import csv
import json
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent.parent / "data" / "bronze"
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "player_predictions.csv"

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

PPG_WEIGHT = 0.6
FORM_WEIGHT = 0.4


def latest_file(prefix: str) -> Path:
    matches = sorted(RAW_DIR.glob(f"{prefix}_*.json"), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No raw file found for prefix '{prefix}' in {RAW_DIR}")
    return matches[0]


def load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def build_team_map(teams: list) -> dict:
    return {t["id"]: t["short_name"] for t in teams}


def chance_as_float(value: int | None) -> float:
    """Convert FPL's chance_of_playing (0-100 or None) to a 0-1 float."""
    if value is None:
        return 1.0
    return value / 100.0


def build_predictions(players: list, team_map: dict) -> list[dict]:
    rows = []
    for p in players:
        # Skip players with no minutes (unreliable PPG) or who are removed
        if p.get("minutes", 0) == 0 or p.get("removed", False):
            continue

        ppg = float(p["points_per_game"])
        form = float(p["form"])
        availability = chance_as_float(p["chance_of_playing_next_round"])

        weighted_score = (PPG_WEIGHT * ppg) + (FORM_WEIGHT * form)
        expected_points = round(weighted_score * availability, 2)

        rows.append({
            "id": p["id"],
            "name": p["web_name"],
            "pos": POSITION_MAP[p["element_type"]],
            "team": team_map[p["team"]],
            "cost": p["now_cost"] / 10,  # convert to £m
            "expected_points": expected_points,
        })

    return sorted(rows, key=lambda r: r["expected_points"], reverse=True)


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "name", "pos", "team", "cost", "expected_points"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    players = load_json(latest_file("elements"))
    teams = load_json(latest_file("teams"))
    team_map = build_team_map(teams)

    predictions = build_predictions(players, team_map)
    save_csv(predictions, OUTPUT_FILE)

    print(f"Saved {len(predictions)} players -> {OUTPUT_FILE}")
    print("\nTop 10 by expected points:")
    print(f"{'Name':<20} {'Pos':<5} {'Team':<5} {'Cost':>6}  {'xPts':>6}")
    print("-" * 48)
    for p in predictions[:10]:
        print(f"{p['name']:<20} {p['pos']:<5} {p['team']:<5} £{p['cost']:>4.1f}m  {p['expected_points']:>6.2f}")


if __name__ == "__main__":
    main()
