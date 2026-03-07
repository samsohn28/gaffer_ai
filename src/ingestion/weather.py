"""Fetch weather data for each FPL fixture from Open-Meteo.

No API key required. Uses archive API for past fixtures, forecast API for future.

Run:
    python -m src.ingestion.weather
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"

STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "Arsenal": (51.5549, -0.1084),
    "Aston Villa": (52.5090, -1.8852),
    "Bournemouth": (50.7352, -1.8376),
    "Brentford": (51.4907, -0.2890),
    "Brighton": (50.8618, -0.0831),
    "Burnley": (53.7890, -2.2302),
    "Chelsea": (51.4816, -0.1909),
    "Crystal Palace": (51.3983, -0.0854),
    "Everton": (53.4388, -2.9669),
    "Fulham": (51.4750, -0.2217),
    "Ipswich": (52.0548, 1.1446),
    "Leeds": (53.7773, -1.5724),
    "Leicester": (52.6203, -1.1424),
    "Liverpool": (53.4308, -2.9609),
    "Man City": (53.4831, -2.2004),
    "Man Utd": (53.4631, -2.2913),
    "Newcastle": (54.9754, -1.6218),
    "Nott'm Forest": (52.9400, -1.1326),
    "Southampton": (50.9058, -1.3914),
    "Spurs": (51.6042, -0.0665),
    "Sunderland": (54.9145, -1.3880),
    "West Ham": (51.5388, -0.0162),
    "Wolves": (52.5904, -2.1309),
}

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,precipitation,windspeed_10m"


def _fetch_weather(lat: float, lon: float, start: str, end: str, use_archive: bool) -> dict:
    url = ARCHIVE_URL if use_archive else FORECAST_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_hourly(data: dict) -> dict[tuple[str, int], dict]:
    """Parse Open-Meteo hourly response → {(date_str, hour): {temp, precip, wind}}"""
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precips = hourly.get("precipitation", [])
    winds = hourly.get("windspeed_10m", [])
    lookup = {}
    for i, t in enumerate(times):
        # t is "YYYY-MM-DDTHH:MM"
        date_str, hr_str = t[:10], t[11:13]
        lookup[(date_str, int(hr_str))] = {
            "temp_c": temps[i] if i < len(temps) else None,
            "precipitation_mm": precips[i] if i < len(precips) else None,
            "windspeed_kmh": winds[i] if i < len(winds) else None,
        }
    return lookup


def main():
    BRONZE.mkdir(parents=True, exist_ok=True)

    fixtures = pd.read_parquet(SILVER / "fixtures.parquet")
    teams = pd.read_parquet(SILVER / "teams.parquet")
    id_to_name = dict(zip(teams["id"], teams["name"]))

    today = date.today()

    # Ensure kickoff_date is a date object
    if "kickoff_date" not in fixtures.columns:
        fixtures["kickoff_date"] = pd.to_datetime(
            fixtures["kickoff_time"], utc=True, errors="coerce"
        ).dt.date

    valid = fixtures.dropna(subset=["kickoff_date", "event", "kickoff_hour"]).copy()
    valid["kickoff_date"] = valid["kickoff_date"].apply(
        lambda d: d if isinstance(d, date) else date.fromisoformat(str(d))
    )

    records = []

    for team_h_id, group in valid.groupby("team_h"):
        team_name = id_to_name.get(team_h_id)
        coords = STADIUM_COORDS.get(team_name)
        if coords is None:
            print(f"  No coords for {team_name} (id={team_h_id}), skipping")
            continue

        lat, lon = coords
        print(f"Fetching weather for {team_name} ({len(group)} home fixtures)...")

        forecast_cutoff = today + timedelta(days=16)

        past = group[group["kickoff_date"] < today]
        future = group[
            (group["kickoff_date"] >= today) & (group["kickoff_date"] <= forecast_cutoff)
        ]

        weather_lookup: dict[tuple[str, int], dict] = {}

        if not past.empty:
            start = str(past["kickoff_date"].min())
            end = str(past["kickoff_date"].max())
            try:
                data = _fetch_weather(lat, lon, start, end, use_archive=True)
                weather_lookup.update(_parse_hourly(data))
                time.sleep(0.1)
            except Exception as e:
                print(f"  Archive fetch failed for {team_name}: {e}")

        if not future.empty:
            start = str(future["kickoff_date"].min())
            end = str(future["kickoff_date"].max())
            try:
                data = _fetch_weather(lat, lon, start, end, use_archive=False)
                weather_lookup.update(_parse_hourly(data))
                time.sleep(0.1)
            except Exception as e:
                print(f"  Forecast fetch failed for {team_name}: {e}")

        for _, fx in group.iterrows():
            date_str = str(fx["kickoff_date"])
            hour = int(fx["kickoff_hour"])
            wx = weather_lookup.get((date_str, hour), {})
            records.append({
                "fixture_id": int(fx["id"]),
                "team_h": int(fx["team_h"]),
                "team_a": int(fx["team_a"]),
                "gw": int(fx["event"]),
                "kickoff_time": str(fx.get("kickoff_time", "")),
                "temp_c": wx.get("temp_c"),
                "precipitation_mm": wx.get("precipitation_mm"),
                "windspeed_kmh": wx.get("windspeed_kmh"),
            })

    out = BRONZE / "weather.json"
    out.write_text(json.dumps(records, indent=2))
    filled = sum(1 for r in records if r["temp_c"] is not None)
    print(f"\nSaved {len(records)} fixture weather records → {out}")
    print(f"Records with weather data: {filled}/{len(records)}")


if __name__ == "__main__":
    main()
