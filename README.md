# GafferAI

An FPL (Fantasy Premier League) AI that picks your squad each gameweek using data-driven predictions and linear programming optimization.

## How it works

1. **Ingest** — fetch live data from the FPL API and xG data from Understat
2. **Score** — estimate expected points per player using heuristics or an ML model
3. **Optimize** — solve for the best 15-man squad + starting XI within FPL constraints

## Quickstart

```bash
python -m src.ingestion.fpl_api          # fetch latest FPL data
python -m src.features.build_heuristics  # score players (heuristic)
python -m src.optimizer.squad_builder    # pick optimal squad
```

## Roadmap

### Data ingestion
- [x] FPL bootstrap-static API (players, teams, gameweeks)
- [ ] FPL live gameweek endpoint (in-game points, bonus)
- [ ] FPL fixtures endpoint (schedule, home/away)
- [ ] Understat scraper (xG, xA, xGI per player per match)
- [ ] Historical season backfill

### Feature engineering
- [x] Heuristic scoring: weighted PPG + form × availability
- [ ] Fixture difficulty rating (FDR) per gameweek
- [ ] Rolling xG / xA features (3 GW, 6 GW, season)
- [ ] Home/away performance splits
- [ ] Minutes played stability (flag rotation risks)
- [ ] Opponent defensive strength
- [ ] Expected Minutes ($xM$) model: predict likelihood of starting vs. subbing
- [ ] Transfer "heat": scrape net transfers in/out to predict price rises (team value protection)
- [ ] Set piece dominance: flag designated penalty/corner takers (major $xP$ boosters)

### Prediction model
- [ ] Train regression model (xPts per player per GW)
- [ ] Cross-validate against historical seasons
- [ ] Calibrate probability of clean sheet by team
- [ ] Predict price changes

### Squad optimizer
- [x] LP optimizer: 15-man squad within £100m budget
- [x] Starting XI selection (formation constraints)
- [x] Captain selection (highest xPts)
- [ ] Transfer optimizer: optimal 1–2 transfers from existing squad
- [ ] Chip strategy: wildcard, bench boost, free hit, triple captain
- [ ] Differential recommendation (low-ownership high-upside picks)
- [ ] Multi-GW horizon: maximize $\sum xP$ over the next 4–6 gameweeks
- [ ] Transfer cost penalty: factor in the $-4$ point hit for exceeding free transfers
- [ ] Bench ordering logic: order subs by $xP$ to maximize auto-sub returns

### Backtesting & evaluation
- [ ] Simulate season GW-by-GW with historical data
- [ ] Score predictions against actual points (MAE, rank correlation)
- [ ] Compare heuristic vs ML model vs random baseline
- [ ] Vegas/bookie baseline: compare picks against anytime goalscorer implied probabilities
- [ ] Point-in-time simulation: ensure no future data leaks into past GW decisions

### Interface
- [ ] CLI: single command to run full pipeline end-to-end
- [ ] Output: formatted squad recommendation with xPts breakdown
- [ ] Export: JSON and CSV squad outputs
- [ ] Web UI or Telegram bot for weekly squad suggestions
