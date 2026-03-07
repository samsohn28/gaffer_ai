"""
Fetch per-GW player history from the FPL element-summary endpoint.

Saves combined history rows for all active players to data/bronze/player_history.json.
Each row = one player × one gameweek with cost, points, ict_index, etc.

Run:
    python -m src.ingestion.backfill_player_history
"""

import json
import sys
import time
from pathlib import Path

import requests

ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
BRONZE = Path(__file__).resolve().parents[2] / "data" / "bronze"


def active_player_ids() -> list[int]:
    """Return FPL IDs of players with minutes > 0 this season."""
    import pandas as pd
    from pathlib import Path
    silver = Path(__file__).resolve().parents[2] / "data" / "silver"
    players = pd.read_parquet(silver / "players.parquet")
    return players.loc[players["minutes"] > 0, "id"].tolist()


def main():
    player_ids = active_player_ids()
    print(f"Fetching element-summary for {len(player_ids)} active players...")

    all_history = []
    errors = []

    for i, pid in enumerate(player_ids, 1):
        url = ELEMENT_SUMMARY_URL.format(player_id=pid)
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  [{i}/{len(player_ids)}] player {pid}: ERROR — {e}", file=sys.stderr)
            errors.append(pid)
            continue

        for row in data.get("history", []):
            all_history.append({
                "player_id": pid,
                "round": row["round"],
                "value": row["value"],
                "total_points": row["total_points"],
                "ict_index": row["ict_index"],
                "minutes": row["minutes"],
                "starts": row["starts"],
                "expected_goals": row.get("expected_goals"),
                "expected_assists": row.get("expected_assists"),
                "expected_goal_involvements": row.get("expected_goal_involvements"),
                "expected_goals_conceded": row.get("expected_goals_conceded"),
                "selected": row.get("selected"),
                "transfers_in": row.get("transfers_in"),
                "transfers_out": row.get("transfers_out"),
            })

        if i % 50 == 0:
            print(f"  {i}/{len(player_ids)} done ({len(all_history)} rows so far)")

        time.sleep(0.2)

    out = BRONZE / "player_history.json"
    out.write_text(json.dumps(all_history))
    print(f"\nSaved {len(all_history)} rows → {out}")
    if errors:
        print(f"Failed player IDs: {errors}", file=sys.stderr)


if __name__ == "__main__":
    main()
