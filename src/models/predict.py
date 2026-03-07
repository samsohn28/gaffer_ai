"""
Generate next-GW expected points predictions using the trained XGBoost model.

Reads next-GW rows from gold/features.parquet, applies the same feature
preprocessing as train.py, and writes player_predictions.csv.

Run:
    python -m src.models.predict
"""

from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from src.models.train import LEAKAGE_COLS, ID_COLS, TARGET, make_features

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "gold" / "features.parquet"
MODEL_PATH = ROOT / "data" / "processed" / "xgb_model.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "player_predictions.csv"


def main():
    df = pd.read_parquet(FEATURES_PATH)
    next_gw = df[df["is_historical"] == False].copy()
    print(f"Next-GW rows: {len(next_gw)}")

    model = XGBRegressor()
    model.load_model(MODEL_PATH)

    # Build feature matrix using same logic as training
    X, _ = make_features(next_gw)

    # Align to model's expected feature columns (handles any get_dummies differences)
    expected_cols = model.get_booster().feature_names
    X = X.reindex(columns=expected_cols, fill_value=0)

    preds = model.predict(X)

    results = next_gw[["player_id", "player_name", "position", "team_name", "cost"]].copy()
    results = results.rename(columns={
        "player_id": "id",
        "player_name": "name",
        "position": "pos",
        "team_name": "team",
    })
    results["expected_points"] = preds.round(2)
    results = results.sort_values("expected_points", ascending=False).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(results)} predictions -> {OUTPUT_PATH}")
    print(f"\nTop 10 by expected points:")
    print(f"{'Name':<22} {'Pos':<5} {'Team':<6} {'Cost':>6}  {'xPts':>6}")
    print("-" * 50)
    for _, p in results.head(10).iterrows():
        print(f"{p['name']:<22} {p['pos']:<5} {p['team']:<6} £{p['cost']:>4.1f}m  {p['expected_points']:>6.2f}")


if __name__ == "__main__":
    main()
