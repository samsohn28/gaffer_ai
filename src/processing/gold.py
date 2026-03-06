"""
Join silver Parquet files → gold/features.parquet.

The gold table has one row per (player_id, gameweek_id):
  - Historical rows (GW1-29): sourced from understat per-match data
  - Next GW row (GW30): sourced from FPL elements snapshot

Run:
    python -m src.processing.gold
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Normalize understat team names → FPL team names (teams.json "name" field)
UNDERSTAT_TO_FPL = {
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Tottenham": "Spurs",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle",
    # Direct matches — included for completeness
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds",
    "Leeds United": "Leeds",
    "Liverpool": "Liverpool",
    "Newcastle": "Newcastle",
    "Sunderland": "Sunderland",
    "West Ham": "West Ham",
    "Wolves": "Wolves",
    "Ipswich": "Ipswich",
    "Ipswich Town": "Ipswich",
    "Leicester": "Leicester",
    "Leicester City": "Leicester",
    "Southampton": "Southampton",
}


def _normalize_understat_team(name: str) -> str:
    return UNDERSTAT_TO_FPL.get(name, name)


def _price_rise_prob(row: pd.Series) -> float:
    net = row["transfers_in_event"] - row["transfers_out_event"]
    selected_pct = row["selected_by_percent"]
    if pd.isna(selected_pct) or selected_pct <= 0:
        return 0.0
    ownership_n = selected_pct / 100 * 11_000_000
    if ownership_n == 0:
        return 0.0
    raw = (net / ownership_n * 100) / 5
    return float(max(0.0, min(1.0, raw)))


def build_player_gw_features(history: pd.DataFrame) -> pd.DataFrame:
    """
    From per-GW player history, compute pre-match rolling features for each
    (player_id, round) combo:
      - cost          : value / 10 at that round
      - ppg           : cumulative points / games *before* this round
      - form          : rolling 5-game points average *before* this round
      - ict_index_form: rolling 5-game ict average *before* this round
    """
    history = history.sort_values(["player_id", "round"]).copy()

    rows = []
    for pid, grp in history.groupby("player_id"):
        grp = grp.sort_values("round").reset_index(drop=True)
        cum_points = 0.0
        cum_games = 0

        for i, row in grp.iterrows():
            rnd = int(row["round"])
            cost = row["value"] / 10.0

            # Pre-match: use stats from all previous rounds
            if cum_games == 0:
                ppg = None
                form = None
                ict_form = None
            else:
                ppg = cum_points / cum_games
                prev = grp.loc[:i - 1]
                tail5 = prev.tail(5)
                form = float(tail5["total_points"].mean())
                ict_form = float(tail5["ict_index"].mean())

            rows.append({
                "player_id": pid,
                "gameweek_id": rnd,
                "cost": cost,
                "ppg": ppg,
                "form": form,
                "ict_index_form": ict_form,
            })

            cum_points += row["total_points"]
            cum_games += 1

    return pd.DataFrame(rows)


def build_historical_rows(
    us_matches: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    fpl_map: pd.DataFrame,
    players: pd.DataFrame,
    live_points: dict[tuple[int, int], int],
    player_gw_features: pd.DataFrame | None = None,
    hist_elo: dict[tuple[int, str], float] | None = None,
) -> pd.DataFrame:
    """Build one row per understat match record, mapped to a GW."""

    # Build FPL team name → id lookup
    name_to_id = dict(zip(teams["name"], teams["id"]))

    # Build fixture lookup: (date_str, fpl_h_name, fpl_a_name) → (gw, team_h_id, team_a_id)
    # kickoff_date is a datetime.date object; stringify to match understat's "YYYY-MM-DD" strings.
    id_to_name = dict(zip(teams["id"], teams["name"]))
    fix_lookup: dict[tuple, tuple] = {}
    for _, fx in fixtures.iterrows():
        if pd.isna(fx["kickoff_date"]) or pd.isna(fx["event"]):
            continue
        h_name = id_to_name.get(fx["team_h"], "")
        a_name = id_to_name.get(fx["team_a"], "")
        key = (str(fx["kickoff_date"]), h_name, a_name)
        fix_lookup[key] = (int(fx["event"]), int(fx["team_h"]), int(fx["team_a"]))

    # Build fpl_map: understat_id (int) → fpl_id
    us_to_fpl = dict(zip(fpl_map["understat_id"], fpl_map["fpl_id"]))

    # Build players lookup: fpl_id → {web_name, element_type, team, is_penalty_taker, is_set_piece_taker}
    pl_lookup = players.set_index("id")[
        ["web_name", "element_type", "team", "is_penalty_taker", "is_set_piece_taker"]
    ].to_dict("index")

    # Build (player_id, gw) → {cost, ppg, form, ict_index_form}
    pgf_lookup: dict[tuple[int, int], dict] = {}
    if player_gw_features is not None and not player_gw_features.empty:
        for _, r in player_gw_features.iterrows():
            pgf_lookup[(int(r["player_id"]), int(r["gameweek_id"]))] = {
                "cost": r["cost"],
                "ppg": r["ppg"],
                "form": r["form"],
                "ict_index_form": r["ict_index_form"],
            }

    rows = []
    for _, m in us_matches.iterrows():
        us_pid = int(m["player_id"])
        fpl_id = us_to_fpl.get(us_pid)
        if fpl_id is None:
            continue

        # Normalize team names for fixture lookup
        h_fpl = _normalize_understat_team(str(m["h_team"]))
        a_fpl = _normalize_understat_team(str(m["a_team"]))
        match_date = m["date"]

        fix = fix_lookup.get((match_date, h_fpl, a_fpl))
        if fix is None:
            continue

        gw, team_h_id, team_a_id = fix
        pl = pl_lookup.get(fpl_id, {})
        pos_code = pl.get("element_type")
        position = POSITION_MAP.get(pos_code, "") if pos_code else ""
        fpl_team_id = pl.get("team")

        is_home = None
        opponent_team_id = None
        if fpl_team_id == team_h_id:
            is_home = True
            opponent_team_id = team_a_id
        elif fpl_team_id == team_a_id:
            is_home = False
            opponent_team_id = team_h_id

        actual_pts = live_points.get((gw, fpl_id))
        pgf = pgf_lookup.get((fpl_id, gw), {})

        rows.append({
            "player_id": fpl_id,
            "gameweek_id": gw,
            "player_name": pl.get("web_name", ""),
            "position": position,
            "team_id": fpl_team_id,
            "team_name": "",  # filled below
            "opponent_team_id": opponent_team_id,
            "is_home": is_home,
            # Cost & ownership
            "cost": pgf.get("cost"),
            "selected_pct": None,
            "transfers_in_event": None,
            "transfers_out_event": None,
            "price_rise_probability": None,
            # Form & scoring
            "ppg": pgf.get("ppg"),
            "form": pgf.get("form"),
            "ict_index_form": pgf.get("ict_index_form"),
            "chance_of_playing": 100,
            "injury_confidence": 100,
            "actual_points": actual_pts,
            # xG / xA match stats
            "xG_match": m.get("xG"),
            "xA_match": m.get("xA"),
            "goals_match": m.get("goals"),
            "assists_match": m.get("assists"),
            "minutes_match": m.get("minutes"),
            "shots_match": m.get("shots"),
            "key_passes_match": m.get("key_passes"),
            # Defensive stats — null for historical
            "defensive_contribution_per90": None,
            "recoveries_per90": None,
            "tackles_per90": None,
            # Set piece flags (static per player)
            "is_penalty_taker": pl.get("is_penalty_taker"),
            "is_set_piece_taker": pl.get("is_set_piece_taker"),
            # Elo — from historical snapshot if available
            "elo_for": hist_elo.get((gw, id_to_name.get(fpl_team_id, ""))) if hist_elo else None,
            "elo_against": hist_elo.get((gw, id_to_name.get(opponent_team_id, ""))) if hist_elo and opponent_team_id else None,
            # Odds — null for historical
            "clean_sheet_prob": None,
            "goalscorer_prob": None,
            "is_historical": True,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Fill team_name from teams
    id_to_short = dict(zip(teams["id"], teams["short_name"]))
    df["team_name"] = df["team_id"].map(id_to_short)

    return df


def build_historical_rows_fpl(
    player_history: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
    player_gw_features: pd.DataFrame | None = None,
    hist_elo: dict[tuple[int, str], float] | None = None,
) -> pd.DataFrame:
    """
    Build historical training rows from player_history (FPL-native data).

    Each row has the same feature profile as a next-GW prediction row:
    cost/ppg/form/ict_index_form from rolling history, is_home/opponent from
    fixtures, elo from historical ClubElo snapshots. Target = total_points.
    """
    id_to_name = dict(zip(teams["id"], teams["name"]))
    id_to_short = dict(zip(teams["id"], teams["short_name"]))

    # player_id → {team_id, element_type, is_penalty_taker, is_set_piece_taker, web_name}
    pl_lookup = players.set_index("id")[
        ["team", "element_type", "is_penalty_taker", "is_set_piece_taker", "web_name"]
    ].to_dict("index")

    # (team_id, gw) → (opponent_team_id, is_home, team_h_difficulty, team_a_difficulty)
    fixture_lookup: dict[tuple[int, int], tuple] = {}
    for _, fx in fixtures.iterrows():
        if pd.isna(fx.get("event")):
            continue
        gw = int(fx["event"])
        h_id, a_id = int(fx["team_h"]), int(fx["team_a"])
        fixture_lookup[(h_id, gw)] = (a_id, True)
        fixture_lookup[(a_id, gw)] = (h_id, False)

    # (player_id, gw) → rolling features
    pgf_lookup: dict[tuple[int, int], dict] = {}
    if player_gw_features is not None and not player_gw_features.empty:
        for _, r in player_gw_features.iterrows():
            pgf_lookup[(int(r["player_id"]), int(r["gameweek_id"]))] = {
                "cost": r["cost"],
                "ppg": r["ppg"],
                "form": r["form"],
                "ict_index_form": r["ict_index_form"],
            }

    rows = []
    for _, ph in player_history.iterrows():
        pid = int(ph["player_id"])
        gw = int(ph["round"])
        actual_pts = ph.get("total_points")
        if actual_pts is None:
            continue

        pl = pl_lookup.get(pid, {})
        fpl_team_id = pl.get("team")
        if fpl_team_id is None:
            continue

        fix = fixture_lookup.get((fpl_team_id, gw))
        if fix is None:
            opponent_team_id, is_home = None, None
        else:
            opponent_team_id, is_home = fix

        own_name = id_to_name.get(fpl_team_id, "")
        opp_name = id_to_name.get(opponent_team_id, "") if opponent_team_id else ""

        pgf = pgf_lookup.get((pid, gw), {})
        pos_code = pl.get("element_type")
        position = POSITION_MAP.get(pos_code, "") if pos_code else ""

        rows.append({
            "player_id": pid,
            "gameweek_id": gw,
            "player_name": pl.get("web_name", ""),
            "position": position,
            "team_id": fpl_team_id,
            "team_name": id_to_short.get(fpl_team_id, ""),
            "opponent_team_id": opponent_team_id,
            "is_home": is_home,
            "cost": pgf.get("cost"),
            "selected_pct": None,
            "transfers_in_event": None,
            "transfers_out_event": None,
            "price_rise_probability": None,
            "ppg": pgf.get("ppg"),
            "form": pgf.get("form"),
            "ict_index_form": pgf.get("ict_index_form"),
            "chance_of_playing": 100,
            "injury_confidence": 100,
            "actual_points": float(actual_pts),
            "xG_match": None,
            "xA_match": None,
            "goals_match": None,
            "assists_match": None,
            "minutes_match": None,
            "shots_match": None,
            "key_passes_match": None,
            "defensive_contribution_per90": None,
            "recoveries_per90": None,
            "tackles_per90": None,
            "is_penalty_taker": pl.get("is_penalty_taker"),
            "is_set_piece_taker": pl.get("is_set_piece_taker"),
            "elo_for": hist_elo.get((gw, own_name)) if hist_elo else None,
            "elo_against": hist_elo.get((gw, opp_name)) if hist_elo and opp_name else None,
            "clean_sheet_prob": None,
            "goalscorer_prob": None,
            "is_historical": True,
        })

    return pd.DataFrame(rows)


def _detect_next_gw() -> int:
    """Return the next GW id using events.json is_next marker, with fallbacks."""
    events_path = BRONZE / "events.json"
    if events_path.exists():
        events = json.loads(events_path.read_text())
        for e in events:
            if e.get("is_next"):
                return e["id"]
        # Fallback: first unfinished event
        for e in events:
            if not e.get("finished"):
                return e["id"]
    raise RuntimeError("Cannot determine next GW from events.json")


def build_next_gw_rows(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    clubelo: pd.DataFrame,
    odds: pd.DataFrame,
    injuries: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per active player for the next upcoming GW."""

    # Filter to players with minutes > 0
    active = players[players["minutes"] > 0].copy()

    # Detect next GW using events.json is_next marker
    next_gw = _detect_next_gw()

    # Next GW fixtures
    next_fixtures = fixtures[fixtures["event"] == next_gw][["team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]].copy()

    # Build team name → elo lookup
    elo_lookup = dict(zip(clubelo["team_name"], clubelo["elo"]))

    # Build id → name lookups
    id_to_name = dict(zip(teams["id"], teams["name"]))
    id_to_short = dict(zip(teams["id"], teams["short_name"]))

    # Build injuries lookup: fpl_id → confidence
    inj_lookup = dict(zip(injuries["fpl_id"], injuries["confidence"]))

    # Build odds lookup: player_name → {clean_sheet_prob, goalscorer_prob}
    odds_lookup: dict[str, dict] = {}
    for _, o in odds.iterrows():
        if int(o["gameweek"]) == next_gw:
            odds_lookup[o["player_name"]] = {
                "clean_sheet_prob": o.get("clean_sheet_prob"),
                "goalscorer_prob": o.get("goalscorer_prob"),
            }

    rows = []
    for _, p in active.iterrows():
        fpl_team_id = int(p["team"])
        mins = p.get("minutes", 0) or 0

        # Find fixture for this team in next GW
        h_fix = next_fixtures[next_fixtures["team_h"] == fpl_team_id]
        a_fix = next_fixtures[next_fixtures["team_a"] == fpl_team_id]

        if not h_fix.empty:
            fx = h_fix.iloc[0]
            is_home = True
            opponent_team_id = int(fx["team_a"])
        elif not a_fix.empty:
            fx = a_fix.iloc[0]
            is_home = False
            opponent_team_id = int(fx["team_h"])
        else:
            # No fixture this GW (blank)
            is_home = None
            opponent_team_id = None

        # Elo
        own_team_name = id_to_name.get(fpl_team_id, "")
        opp_team_name = id_to_name.get(opponent_team_id, "") if opponent_team_id else ""
        elo_for = elo_lookup.get(own_team_name)
        elo_against = elo_lookup.get(opp_team_name) if opp_team_name else None

        # Odds — match on web_name
        web_name = p["web_name"]
        o_entry = odds_lookup.get(web_name, {})

        # Injury confidence
        inj_conf = inj_lookup.get(int(p["id"]), 100)

        # Per-90 defensive stats
        mins_90 = mins / 90 if mins > 0 else None
        def_per90 = (p.get("defensive_contribution") / mins_90) if mins_90 else None
        rec_per90 = (p.get("recoveries") / mins_90) if mins_90 else None
        tac_per90 = (p.get("tackles") / mins_90) if mins_90 else None

        # Price rise probability
        selected_pct = p.get("selected_by_percent")
        xin = p.get("transfers_in_event", 0) or 0
        xout = p.get("transfers_out_event", 0) or 0
        if selected_pct and selected_pct > 0:
            ownership_n = selected_pct / 100 * 11_000_000
            net = xin - xout
            rise_prob = float(max(0.0, min(1.0, (net / ownership_n * 100) / 5)))
        else:
            rise_prob = 0.0

        pos_code = p.get("element_type")
        position = POSITION_MAP.get(pos_code, "") if pos_code else ""

        rows.append({
            "player_id": int(p["id"]),
            "gameweek_id": next_gw,
            "player_name": web_name,
            "position": position,
            "team_id": fpl_team_id,
            "team_name": id_to_short.get(fpl_team_id, ""),
            "opponent_team_id": opponent_team_id,
            "is_home": is_home,
            # Cost & ownership
            "cost": p["now_cost"] / 10,
            "selected_pct": selected_pct,
            "transfers_in_event": int(xin),
            "transfers_out_event": int(xout),
            "price_rise_probability": rise_prob,
            # Form & scoring
            "ppg": p.get("points_per_game"),
            "form": p.get("form"),
            "ict_index_form": p.get("ict_index"),
            "chance_of_playing": int(p.get("chance_of_playing_next_round", 100)),
            "injury_confidence": int(inj_conf),
            "actual_points": None,
            # xG / xA match stats — null (no match played yet)
            "xG_match": None,
            "xA_match": None,
            "goals_match": None,
            "assists_match": None,
            "minutes_match": None,
            "shots_match": None,
            "key_passes_match": None,
            # Defensive stats
            "defensive_contribution_per90": def_per90,
            "recoveries_per90": rec_per90,
            "tackles_per90": tac_per90,
            # Set piece flags
            "is_penalty_taker": bool(p.get("is_penalty_taker", False)),
            "is_set_piece_taker": bool(p.get("is_set_piece_taker", False)),
            # Elo
            "elo_for": elo_for,
            "elo_against": elo_against,
            # Odds
            "clean_sheet_prob": o_entry.get("clean_sheet_prob"),
            "goalscorer_prob": o_entry.get("goalscorer_prob"),
            "is_historical": False,
        })

    return pd.DataFrame(rows)


def main():
    GOLD.mkdir(parents=True, exist_ok=True)

    print("Loading silver tables...")
    players = pd.read_parquet(SILVER / "players.parquet")
    teams = pd.read_parquet(SILVER / "teams.parquet")
    fixtures = pd.read_parquet(SILVER / "fixtures.parquet")
    clubelo = pd.read_parquet(SILVER / "clubelo.parquet")
    odds = pd.read_parquet(SILVER / "odds.parquet")
    injuries = pd.read_parquet(SILVER / "injuries.parquet")

    # Load all live GW files → (gw, fpl_id) → total_points
    live_points: dict[tuple[int, int], int] = {}
    live_files = sorted(BRONZE.glob("live_gw_*.json"))
    for lf in live_files:
        try:
            gw_num = int(lf.stem.split("_")[-1])
        except ValueError:
            continue
        data = json.loads(lf.read_text())
        for el in data.get("elements", []):
            pts = el["stats"].get("total_points")
            if pts is not None:
                live_points[(gw_num, el["id"])] = pts
    print(f"Loaded {len(live_files)} live GW file(s), {len(live_points)} (gw, player) point entries")

    ph = None
    player_gw_features = None
    ph_path = SILVER / "player_history.parquet"
    if ph_path.exists():
        ph = pd.read_parquet(ph_path)
        player_gw_features = build_player_gw_features(ph)
        print(f"Built per-GW rolling features for {player_gw_features['player_id'].nunique()} players")
    else:
        print("player_history.parquet not found — cost/ppg/form will be null for historical rows")

    hist_elo: dict[tuple[int, str], float] | None = None
    hist_elo_path = SILVER / "clubelo_historical.parquet"
    if hist_elo_path.exists():
        ch = pd.read_parquet(hist_elo_path)
        hist_elo = {(int(r["gw"]), r["team_name"]): r["elo"] for _, r in ch.iterrows()}
        print(f"Loaded historical Elo for {ch['gw'].nunique()} GWs, {len(hist_elo)} (gw, team) entries")
    else:
        print("clubelo_historical.parquet not found — elo_for/against will be null for historical rows")

    print("Building historical rows (FPL-profile)...")
    if ph is not None:
        hist = build_historical_rows_fpl(ph, fixtures, teams, players, player_gw_features, hist_elo)
    else:
        hist = pd.DataFrame()
    print(f"  FPL historical rows: {len(hist)}")

    # Enrich with Understat xG/xA where available
    fpl_map_path = SILVER / "fpl_map.parquet"
    us_matches_path = BRONZE / "understat_matches_2025.json"
    if fpl_map_path.exists() and us_matches_path.exists() and not hist.empty:
        print("Enriching with Understat xG/xA...")
        fpl_map = pd.read_parquet(fpl_map_path)
        us_matches_raw = json.loads(us_matches_path.read_text())
        us_matches = pd.DataFrame(us_matches_raw)
        us_rows = build_historical_rows(
            us_matches, fixtures, teams, fpl_map, players,
            live_points, player_gw_features, hist_elo,
        )
        print(f"  Understat rows matched: {len(us_rows)}")
        if not us_rows.empty:
            # Aggregate per (player_id, gameweek_id) to handle double GWs
            xg_cols = ["xG_match", "xA_match", "goals_match", "assists_match",
                       "minutes_match", "shots_match", "key_passes_match"]
            us_agg = (
                us_rows.groupby(["player_id", "gameweek_id"])[xg_cols]
                .sum()
                .reset_index()
            )
            # Left-join onto FPL rows and fill nulls
            hist = hist.merge(us_agg, on=["player_id", "gameweek_id"], how="left", suffixes=("", "_us"))
            for col in xg_cols:
                us_col = col + "_us"
                if us_col in hist.columns:
                    hist[col] = hist[col].combine_first(hist[us_col])
                    hist.drop(columns=[us_col], inplace=True)
            matched_rows = hist[xg_cols[0]].notna().sum()
            print(f"  Rows enriched with xG/xA: {matched_rows}/{len(hist)}")
    else:
        print("Skipping Understat enrichment (fpl_map or understat_matches_2025.json not found)")

    print("Building next GW rows...")
    next_gw = build_next_gw_rows(players, fixtures, teams, clubelo, odds, injuries)
    print(f"  Next GW rows: {len(next_gw)}")

    features = pd.concat([hist, next_gw], ignore_index=True)
    features = features.sort_values(["player_id", "gameweek_id"]).reset_index(drop=True)

    out = GOLD / "features.parquet"
    features.to_parquet(out, index=False)
    print(f"\nWritten: {out}")
    print(f"Total rows: {len(features)}")
    print(f"Columns: {list(features.columns)}")


if __name__ == "__main__":
    main()
