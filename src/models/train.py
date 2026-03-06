"""Train XGBRegressor to predict actual_points from pre-match features."""

from pathlib import Path

import pandas as pd
import numpy as np
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "gold" / "features.parquet"
MODEL_PATH = ROOT / "data" / "processed" / "xgb_model.json"

LEAKAGE_COLS = [
    "xG_match", "xA_match", "goals_match", "assists_match",
    "minutes_match", "shots_match", "key_passes_match",
]
ID_COLS = ["player_id", "player_name", "team_id", "team_name"]
TARGET = "actual_points"


def load_and_prepare(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df[TARGET].notna()].copy()
    return df


def make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop = LEAKAGE_COLS + ID_COLS + [TARGET, "gameweek_id"]
    X = df.drop(columns=[c for c in drop if c in df.columns])

    # One-hot encode position
    if "position" in X.columns:
        X = pd.get_dummies(X, columns=["position"], drop_first=False)

    # Coerce any remaining object columns to numeric (e.g. is_home, prob fields)
    obj_cols = X.select_dtypes(include="object").columns
    if len(obj_cols):
        X[obj_cols] = X[obj_cols].apply(pd.to_numeric, errors="coerce")

    y = df[TARGET]
    return X, y


def time_series_split(df: pd.DataFrame):
    gws = sorted(df["gameweek_id"].unique())
    n = len(gws)

    # Need at least 3 distinct GWs for a meaningful split; otherwise train-only
    if n < 3:
        train_gws = set(gws)
        val_gws: set = set()
        test_gws: set = set()
    else:
        p70_idx = int(n * 0.70)
        p85_idx = int(n * 0.85)
        # Ensure each split has at least one GW
        p70_idx = max(1, p70_idx)
        p85_idx = max(p70_idx + 1, p85_idx)
        train_gws = set(gws[:p70_idx])
        val_gws = set(gws[p70_idx:p85_idx])
        test_gws = set(gws[p85_idx:])

    train_mask = df["gameweek_id"].isin(train_gws)
    val_mask = df["gameweek_id"].isin(val_gws)
    test_mask = df["gameweek_id"].isin(test_gws)
    return train_mask, val_mask, test_mask


def main():
    df = load_and_prepare(FEATURES_PATH)
    n = len(df)
    print(f"Labeled rows: {n}")

    if n < 50:
        print("Insufficient labeled data to train (need ≥ 50 rows). "
              "Fix the understat season mismatch and re-run silver/gold pipelines.")
        return

    train_mask, val_mask, test_mask = time_series_split(df)
    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    def gw_range(d: pd.DataFrame) -> str:
        if d.empty:
            return "—"
        lo, hi = d["gameweek_id"].min(), d["gameweek_id"].max()
        return f"GW{lo}-{hi}" if lo != hi else f"GW{lo}"

    print(
        f"Train {gw_range(train_df)} ({len(train_df)} rows) | "
        f"Val {gw_range(val_df)} ({len(val_df)} rows) | "
        f"Test {gw_range(test_df)} ({len(test_df)} rows)"
    )

    if val_df.empty or test_df.empty:
        print(
            "Not enough distinct gameweeks to split into train/val/test "
            f"(only {df['gameweek_id'].nunique()} GW(s) with labeled data). "
            "Training on all labeled rows without evaluation."
        )

    X_train, y_train = make_features(train_df)
    X_val, y_val = make_features(val_df)
    X_test, y_test = make_features(test_df)

    # Align columns (get_dummies may produce different sets across splits)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # Use early stopping when a validation set is available so the number of
    # trees is chosen empirically rather than fixed. Fall back to a capped
    # n_estimators when there is no val split.
    has_val = not val_df.empty
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        tree_method="hist",
        random_state=42,
        early_stopping_rounds=30 if has_val else None,
        eval_metric="mae" if has_val else None,
    )
    eval_set = [(X_val, y_val)] if has_val else []
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    if has_val:
        print(f"Best iteration: {model.best_iteration}")

    from sklearn.metrics import mean_absolute_error
    if has_val and not test_df.empty:
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        test_mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"Val MAE: {val_mae:.3f}  |  Test MAE: {test_mae:.3f}")
    else:
        train_mae = mean_absolute_error(y_train, model.predict(X_train))
        print(f"Train MAE (no hold-out splits): {train_mae:.3f}")

    importances = pd.Series(
        model.get_booster().get_score(importance_type="gain"),
        name="gain",
    ).sort_values(ascending=False)
    print("\nTop 10 features (gain):")
    print(importances.head(10).to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
