"""
Suggest optimal FPL transfers given a manager's current squad.

Fetches current squad from FPL API, loads next-GW predictions, then uses LP
to find the best 1 or 2 transfers. Shows net expected points gain after any
hit penalty (-4 pts per transfer beyond free transfers).

Usage:
    python -m src.optimizer.transfer_optimizer --team-id 123456
    python -m src.optimizer.transfer_optimizer --team-id 123456 --free-transfers 2
    python -m src.optimizer.transfer_optimizer --team-id 123456 --free-transfers 1 --max-hits 1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pulp
import requests

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
PREDICTIONS_FILE = ROOT / "data" / "processed" / "player_predictions.csv"

HIT_PENALTY = 4.0
MAX_PER_TEAM = 3
POSITION_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


# ── FPL API ────────────────────────────────────────────────────────────────────

def current_gw() -> int:
    events = json.loads((BRONZE / "events.json").read_text())
    for e in events:
        if e.get("is_current"):
            return e["id"]
    finished = [e for e in events if e.get("finished")]
    return finished[-1]["id"] if finished else 1


def fetch_picks(team_id: int, gw: int) -> dict:
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Data loading ───────────────────────────────────────────────────────────────

def load_predictions() -> dict[int, dict]:
    """Return {fpl_id: player_dict} from player_predictions.csv."""
    players = {}
    with open(PREDICTIONS_FILE) as f:
        for row in csv.DictReader(f):
            pid = int(row["id"])
            players[pid] = {
                "id": pid,
                "name": row["name"],
                "pos": row["pos"],
                "team": row["team"],
                "cost": float(row["cost"]),
                "expected_points": float(row["expected_points"]),
            }
    return players


def build_current_squad(picks: list[dict], predictions: dict[int, dict]) -> list[dict]:
    """Map FPL picks to prediction data. Warn on any unmapped players."""
    squad = []
    for pick in picks:
        pid = pick["element"]
        if pid not in predictions:
            print(f"  Warning: player {pid} not in predictions (may be inactive), skipping.")
            continue
        squad.append(predictions[pid])
    return squad


# ── LP optimiser ───────────────────────────────────────────────────────────────

def solve_with_transfers(
    all_players: list[dict],
    current_ids: set[int],
    available_budget: float,
    max_out: int,
) -> list[dict]:
    """
    Return optimal 15-man squad allowing at most `max_out` players sold
    from `current_ids`, within `available_budget`.
    """
    prob = pulp.LpProblem("fpl_transfers", pulp.LpMaximize)

    x = {p["id"]: pulp.LpVariable(f"x_{p['id']}", cat="Binary") for p in all_players}
    keep = {pid: pulp.LpVariable(f"keep_{pid}", cat="Binary") for pid in current_ids}

    # Objective
    prob += pulp.lpSum(p["expected_points"] * x[p["id"]] for p in all_players)

    # Squad size
    prob += pulp.lpSum(x.values()) == 15

    # Position quotas
    for pos, quota in POSITION_QUOTAS.items():
        prob += pulp.lpSum(x[p["id"]] for p in all_players if p["pos"] == pos) == quota

    # Max 3 per club
    teams = {p["team"] for p in all_players}
    for team in teams:
        prob += pulp.lpSum(x[p["id"]] for p in all_players if p["team"] == team) <= MAX_PER_TEAM

    # Budget
    prob += pulp.lpSum(p["cost"] * x[p["id"]] for p in all_players) <= available_budget

    # Keep constraints: keep[i] <= x[i], and kept players must be selected
    for pid in current_ids:
        if pid in x:
            prob += keep[pid] <= x[pid]

    # Transfer limit: number of current players sold <= max_out
    prob += pulp.lpSum(1 - keep[pid] for pid in current_ids if pid in x) <= max_out

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Transfer solver failed: {pulp.LpStatus[prob.status]}")

    return [p for p in all_players if pulp.value(x[p["id"]]) == 1]


# ── Output helpers ─────────────────────────────────────────────────────────────

def squad_xpts(squad: list[dict]) -> float:
    return sum(p["expected_points"] for p in squad)


def diff_squads(current: list[dict], new: list[dict]) -> tuple[list[dict], list[dict]]:
    current_ids = {p["id"] for p in current}
    new_ids = {p["id"] for p in new}
    sold = [p for p in current if p["id"] not in new_ids]
    bought = [p for p in new if p["id"] not in current_ids]
    return sold, bought


def print_transfers(
    transfers_out: list[dict],
    transfers_in: list[dict],
    xpts_gain: float,
    net_gain: float,
    hits: int,
) -> None:
    if not transfers_out:
        print("  No transfers recommended.")
        return
    for out, inn in zip(transfers_out, transfers_in):
        print(f"  OUT  {out['name']:<22} {out['pos']:<5} {out['team']:<6} £{out['cost']:.1f}m  "
              f"xPts: {out['expected_points']:.2f}")
        print(f"  IN   {inn['name']:<22} {inn['pos']:<5} {inn['team']:<6} £{inn['cost']:.1f}m  "
              f"xPts: {inn['expected_points']:.2f}")
        if len(transfers_out) > 1:
            print()
    if hits > 0:
        print(f"  Hit penalty: -{hits * HIT_PENALTY:.0f} pts ({hits} hit(s))")
    print(f"  xPts gain (gross): +{xpts_gain:.2f}")
    print(f"  xPts gain (net):   +{net_gain:.2f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--team-id", type=int, help="FPL manager team ID (fetches squad from API)")
    group.add_argument("--squad-ids", type=str, help="Comma-separated FPL player IDs for current squad")
    parser.add_argument("--bank", type=float, default=0.0, help="Money in bank in £m (default: 0.0)")
    parser.add_argument("--free-transfers", type=int, default=1, help="Free transfers available (default: 1)")
    parser.add_argument("--max-hits", type=int, default=1, help="Max additional hit transfers to consider (default: 1)")
    args = parser.parse_args()

    gw = current_gw()
    predictions = load_predictions()

    if args.squad_ids:
        ids = [int(i.strip()) for i in args.squad_ids.split(",")]
        current_squad = []
        for pid in ids:
            if pid not in predictions:
                print(f"  Warning: player {pid} not in predictions, skipping.")
                continue
            current_squad.append(predictions[pid])
        squad_value = sum(p["cost"] for p in current_squad)
        bank = args.bank
        available_budget = squad_value + bank
        print(f"Current GW: {gw}  |  Free transfers: {args.free_transfers}")
        print(f"Squad value: £{squad_value:.1f}m  |  Bank: £{bank:.1f}m  |  Budget: £{available_budget:.1f}m")
    else:
        print(f"Current GW: {gw}  |  Team ID: {args.team_id}  |  Free transfers: {args.free_transfers}")
        print("Fetching current squad from FPL API...")
        try:
            data = fetch_picks(args.team_id, gw)
        except requests.RequestException as e:
            print(f"Error fetching team: {e}", file=sys.stderr)
            sys.exit(1)
        history = data["entry_history"]
        bank = history["bank"] / 10
        squad_value = history["value"] / 10
        available_budget = bank + squad_value
        print(f"Squad value: £{squad_value:.1f}m  |  Bank: £{bank:.1f}m  |  Budget: £{available_budget:.1f}m")
        current_squad = build_current_squad(data["picks"], predictions)
    current_ids = {p["id"] for p in current_squad}
    all_players = list(predictions.values())

    baseline_xpts = squad_xpts(current_squad)
    print(f"Current squad xPts: {baseline_xpts:.2f}\n")

    print("=" * 60)

    best_net = 0.0
    best_scenario = None

    for extra_hits in range(args.max_hits + 1):
        max_out = args.free_transfers + extra_hits
        hits = max(0, extra_hits)
        label = (f"{max_out}-transfer" + (f" ({hits} hit)" if hits else ""))

        try:
            new_squad = solve_with_transfers(all_players, current_ids, available_budget, max_out)
        except RuntimeError as e:
            print(f"{label}: solver error — {e}")
            continue

        sold, bought = diff_squads(current_squad, new_squad)
        gross_gain = squad_xpts(new_squad) - baseline_xpts
        penalty = hits * HIT_PENALTY
        net_gain = gross_gain - penalty

        print(f"\n  {label.upper()}")
        print_transfers(sold, bought, gross_gain, net_gain, hits)

        if net_gain > best_net:
            best_net = net_gain
            best_scenario = (label, sold, bought, gross_gain, net_gain, hits)

    print(f"\n{'=' * 60}")
    if best_scenario:
        label, sold, bought, gross, net, hits = best_scenario
        print(f"\n  RECOMMENDATION: {label}")
        for out, inn in zip(sold, bought):
            print(f"  {out['name']} → {inn['name']}  (+{inn['expected_points'] - out['expected_points']:.2f} xPts)")
        print(f"  Net xPts gain: +{net:.2f}")
    else:
        print("\n  RECOMMENDATION: Hold — no transfer improves expected points after penalties.")


if __name__ == "__main__":
    main()
