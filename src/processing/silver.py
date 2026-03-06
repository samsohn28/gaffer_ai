"""
Clean bronze JSON → silver Parquet files.

One cleaner per source; each reads from data/bronze/ and writes a single
.parquet to data/silver/.

Run:
    python -m src.processing.silver
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"


def clean_players() -> Path:
    data = json.loads((BRONZE / "elements.json").read_text())
    df = pd.DataFrame(data)

    df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce")
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")

    df["chance_of_playing_next_round"] = df["chance_of_playing_next_round"].fillna(100).astype(int)

    df["is_penalty_taker"] = df["penalties_order"] == 1
    df["is_set_piece_taker"] = (
        (df["corners_and_indirect_freekicks_order"] == 1)
        | (df["direct_freekicks_order"] == 1)
    )

    cols = [
        "id", "web_name", "element_type", "team", "now_cost",
        "points_per_game", "form", "ict_index", "selected_by_percent",
        "transfers_in_event", "transfers_out_event",
        "chance_of_playing_next_round", "minutes",
        "clearances_blocks_interceptions", "recoveries", "tackles",
        "defensive_contribution", "starts",
        "is_penalty_taker", "is_set_piece_taker",
        "removed",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    out = SILVER / "players.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_teams() -> Path:
    data = json.loads((BRONZE / "teams.json").read_text())
    df = pd.DataFrame(data)
    cols = [
        "id", "name", "short_name", "strength",
        "strength_overall_home", "strength_overall_away",
        "strength_attack_home", "strength_attack_away",
        "strength_defence_home", "strength_defence_away",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    out = SILVER / "teams.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_fixtures() -> Path:
    data = json.loads((BRONZE / "fixtures.json").read_text())
    df = pd.DataFrame(data)

    kt = pd.to_datetime(df["kickoff_time"], utc=True, errors="coerce")
    df["kickoff_date"] = kt.dt.date
    df["kickoff_hour"] = kt.dt.hour

    cols = [
        "id", "event", "team_h", "team_a", "finished",
        "kickoff_time", "kickoff_date", "kickoff_hour",
        "team_h_score", "team_a_score",
        "team_h_difficulty", "team_a_difficulty",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    out = SILVER / "fixtures.parquet"
    df.to_parquet(out, index=False)
    return out


def _latest_bronze(pattern: str) -> Path:
    """Return the most recently modified file matching a glob pattern in BRONZE."""
    matches = sorted(BRONZE.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {BRONZE}")
    return matches[0]


def clean_understat_players() -> Path:
    data = json.loads(_latest_bronze("understat_players_*.json").read_text())
    df = pd.DataFrame(data)

    for col in ["xG", "xA", "npxG"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["games", "goals", "assists", "shots", "key_passes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # "time" = season total minutes in understat season-level data
    if "time" in df.columns:
        df = df.rename(columns={"time": "minutes"})
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    cols = ["id", "player_name", "team_title", "games", "minutes", "goals",
            "xG", "assists", "xA", "npxG", "shots", "key_passes"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    out = SILVER / "understat_players.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_understat_matches() -> Path:
    data = json.loads(_latest_bronze("understat_matches_*.json").read_text())
    df = pd.DataFrame(data)

    for col in ["xG", "xA", "npxG", "npg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["goals", "assists", "shots", "key_passes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # "time" = minutes played in this match
    if "time" in df.columns:
        df = df.rename(columns={"time": "minutes"})
        df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    cols = ["player_id", "date", "h_team", "a_team", "minutes",
            "goals", "xG", "assists", "xA", "shots", "key_passes"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    out = SILVER / "understat_matches.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_fpl_map() -> Path:
    data = json.loads((BRONZE / "understat_fpl_map.json").read_text())
    df = pd.DataFrame(data)
    df = df.dropna(subset=["understat_id"])
    df["understat_id"] = df["understat_id"].astype(int)
    cols = ["fpl_id", "understat_id", "fpl_name", "understat_name"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    out = SILVER / "fpl_map.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_clubelo() -> Path:
    data = json.loads((BRONZE / "clubelo.json").read_text())
    ratings = data["ratings"]
    rows = [{"team_name": name, "elo": float(elo)} for name, elo in ratings.items()]
    df = pd.DataFrame(rows)
    out = SILVER / "clubelo.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_odds() -> Path:
    odds_files = sorted(BRONZE.glob("odds_gw_*.json"))
    if not odds_files:
        raise FileNotFoundError("No odds_gw_*.json found in data/bronze/")
    data = json.loads(odds_files[-1].read_text())
    gw = data["gameweek"]

    rows = []
    for fixture in data.get("fixtures", []):
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        cs = fixture.get("clean_sheet_prob", {})
        cs_home = cs.get("home")
        cs_away = cs.get("away")

        for entry in fixture.get("goalscorer_probs", []):
            player_name = entry["player_name"]
            if player_name in ("Yes", "No"):
                continue
            team = entry.get("team", "")
            # Assign clean sheet prob based on which side the player's team is
            if team and home_team and (team == home_team or home_team in team or team in home_team):
                cs_prob = cs_home
            elif team and away_team and (team == away_team or away_team in team or team in away_team):
                cs_prob = cs_away
            else:
                cs_prob = None

            rows.append({
                "gameweek": gw,
                "team": team,
                "player_name": player_name,
                "clean_sheet_prob": cs_prob,
                "goalscorer_prob": entry["prob"],
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["gameweek", "team", "player_name", "clean_sheet_prob", "goalscorer_prob"]
    )
    out = SILVER / "odds.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_player_history() -> Path:
    data = json.loads((BRONZE / "player_history.json").read_text())
    df = pd.DataFrame(data)
    df["ict_index"] = pd.to_numeric(df["ict_index"], errors="coerce")
    cols = ["player_id", "round", "value", "total_points", "ict_index", "minutes", "starts"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    out = SILVER / "player_history.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_clubelo_historical() -> Path:
    data = json.loads((BRONZE / "clubelo_historical.json").read_text())
    rows = []
    for entry in data:
        gw = int(entry["gw"])
        for team_name, elo in entry["ratings"].items():
            rows.append({"gw": gw, "team_name": team_name, "elo": float(elo)})
    df = pd.DataFrame(rows)
    out = SILVER / "clubelo_historical.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_injuries() -> Path:
    data = json.loads((BRONZE / "injuries.json").read_text())
    df = pd.DataFrame(data.get("injuries", []))
    df = df.dropna(subset=["fpl_id"])
    df["fpl_id"] = df["fpl_id"].astype(int)
    cols = ["fpl_id", "player_name", "team", "condition", "confidence"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    out = SILVER / "injuries.parquet"
    df.to_parquet(out, index=False)
    return out


def main():
    SILVER.mkdir(parents=True, exist_ok=True)

    cleaners = [
        ("players.parquet", clean_players),
        ("teams.parquet", clean_teams),
        ("fixtures.parquet", clean_fixtures),
        ("understat_players.parquet", clean_understat_players),
        ("understat_matches.parquet", clean_understat_matches),
        ("fpl_map.parquet", clean_fpl_map),
        ("clubelo.parquet", clean_clubelo),
        ("odds.parquet", clean_odds),
        ("injuries.parquet", clean_injuries),
        ("player_history.parquet", clean_player_history),
        ("clubelo_historical.parquet", clean_clubelo_historical),
    ]

    for name, fn in cleaners:
        try:
            out = fn()
            df = pd.read_parquet(out)
            print(f"{name}: {len(df)} rows")
        except Exception as e:
            print(f"{name}: ERROR - {e}")


if __name__ == "__main__":
    main()
