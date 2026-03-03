"""
Optimize a 15-man FPL squad using linear programming (PuLP).

Squad constraints:
  - Total cost <= £100m
  - Exactly 2 GKPs, 5 DEFs, 5 MIDs, 3 FWDs
  - Max 3 players from any single club
  - Maximize total expected_points

Starting XI constraints (selected from the 15):
  - Exactly 11 starters
  - Exactly 1 GKP
  - At least 3 DEFs, at least 2 MIDs, at least 1 FWD (standard FPL formation rules)
  - Maximize starting XI expected_points (captain counts double)

Captain: the starter with the highest expected_points.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pulp

PREDICTIONS_FILE = Path(__file__).parent.parent.parent / "data" / "processed" / "player_predictions.csv"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

BUDGET = 100.0
POSITION_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3


def load_players(path: Path) -> list[dict]:
    with open(path) as f:
        reader = csv.DictReader(f)
        players = []
        for row in reader:
            players.append({
                "id": int(row["id"]),
                "name": row["name"],
                "pos": row["pos"],
                "team": row["team"],
                "cost": float(row["cost"]),
                "expected_points": float(row["expected_points"]),
            })
    return players


def solve_squad(players: list[dict]) -> list[dict]:
    """Pick the optimal 15-man squad from all available players."""
    prob = pulp.LpProblem("fpl_squad_builder", pulp.LpMaximize)
    selected = [pulp.LpVariable(f"x_{p['id']}", cat="Binary") for p in players]

    prob += pulp.lpSum(p["expected_points"] * x for p, x in zip(players, selected))
    prob += pulp.lpSum(p["cost"] * x for p, x in zip(players, selected)) <= BUDGET
    prob += pulp.lpSum(selected) == 15

    for pos, quota in POSITION_QUOTAS.items():
        prob += pulp.lpSum(x for p, x in zip(players, selected) if p["pos"] == pos) == quota

    teams = {p["team"] for p in players}
    for team in teams:
        prob += pulp.lpSum(x for p, x in zip(players, selected) if p["team"] == team) <= MAX_PER_TEAM

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Squad solver failed: {pulp.LpStatus[prob.status]}")

    return [p for p, x in zip(players, selected) if pulp.value(x) == 1]


def solve_starting_xi(squad: list[dict]) -> list[dict]:
    """
    Pick the best 11 starters from the 15-man squad.

    Formation rules (standard FPL):
      - Exactly 1 GKP
      - At least 3 DEFs
      - At least 2 MIDs
      - At least 1 FWD
      - Exactly 11 starters total
    """
    prob = pulp.LpProblem("fpl_starting_xi", pulp.LpMaximize)
    starting = [pulp.LpVariable(f"s_{p['id']}", cat="Binary") for p in squad]

    prob += pulp.lpSum(p["expected_points"] * s for p, s in zip(squad, starting))
    prob += pulp.lpSum(starting) == 11
    prob += pulp.lpSum(s for p, s in zip(squad, starting) if p["pos"] == "GKP") == 1
    prob += pulp.lpSum(s for p, s in zip(squad, starting) if p["pos"] == "DEF") >= 3
    prob += pulp.lpSum(s for p, s in zip(squad, starting) if p["pos"] == "MID") >= 2
    prob += pulp.lpSum(s for p, s in zip(squad, starting) if p["pos"] == "FWD") >= 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Starting XI solver failed: {pulp.LpStatus[prob.status]}")

    return [p for p, s in zip(squad, starting) if pulp.value(s) == 1]


def pick_captain(starters: list[dict]) -> dict:
    return max(starters, key=lambda p: p["expected_points"])


def print_squad(squad: list[dict], starters: list[dict], captain: dict) -> None:
    starter_ids = {p["id"] for p in starters}
    order = ["GKP", "DEF", "MID", "FWD"]

    total_cost = sum(p["cost"] for p in squad)
    xi_xpts = sum(p["expected_points"] for p in starters) + captain["expected_points"]  # captain doubles

    print(f"\n{'=' * 58}")
    print(f"  GAFFER AI — OPTIMIZED SQUAD")
    print(f"{'=' * 58}")

    print(f"\n  STARTING XI")
    for pos in order:
        for p in sorted(starters, key=lambda p: p["expected_points"], reverse=True):
            if p["pos"] != pos:
                continue
            tag = " (C)" if p["id"] == captain["id"] else "     "
            print(f"  {tag}  {p['name']:<22} {p['team']:<5} £{p['cost']:>4.1f}m  xPts: {p['expected_points']:>5.2f}")

    print(f"\n  BENCH")
    bench = [p for p in squad if p["id"] not in starter_ids]
    for pos in order:
        for p in sorted(bench, key=lambda p: p["expected_points"], reverse=True):
            if p["pos"] != pos:
                continue
            print(f"         {p['name']:<22} {p['team']:<5} £{p['cost']:>4.1f}m  xPts: {p['expected_points']:>5.2f}")

    print(f"\n{'=' * 58}")
    print(f"  Total squad cost:  £{total_cost:.1f}m  (budget: £{BUDGET:.1f}m)")
    print(f"  Starting XI xPts:  {xi_xpts:.2f}  (incl. captain bonus)")
    print(f"  Captain:           {captain['name']}")
    print(f"{'=' * 58}\n")


def save_json(squad: list[dict], starters: list[dict], captain: dict, output_dir: Path) -> Path:
    starter_ids = {p["id"] for p in starters}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"squad_{timestamp}.json"

    payload = {
        "generated_at": timestamp,
        "budget": BUDGET,
        "total_cost": round(sum(p["cost"] for p in squad), 1),
        "captain": captain["name"],
        "starting_xi_expected_points": round(
            sum(p["expected_points"] for p in starters) + captain["expected_points"], 2
        ),
        "starting_xi": [
            {**p, "is_captain": p["id"] == captain["id"]}
            for p in sorted(starters, key=lambda p: p["expected_points"], reverse=True)
        ],
        "bench": [
            p for p in sorted(squad, key=lambda p: p["expected_points"], reverse=True)
            if p["id"] not in starter_ids
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


def main():
    players = load_players(PREDICTIONS_FILE)
    print(f"Loaded {len(players)} players. Solving squad...")

    squad = solve_squad(players)
    starters = solve_starting_xi(squad)
    captain = pick_captain(starters)

    print_squad(squad, starters, captain)

    path = save_json(squad, starters, captain, OUTPUT_DIR)
    print(f"Squad saved -> {path}")


if __name__ == "__main__":
    main()
