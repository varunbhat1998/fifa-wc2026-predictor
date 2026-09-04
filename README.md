# FIFA World Cup 2026 Predictor

Predicts every match of the 2026 FIFA World Cup — pre-match scoreline, win/draw/loss probabilities, full-tournament bracket, and daily tip-optimal picks — with automatic retraining after every result. Delivered to Telegram.

Built to optimise the [WC Tip Game 2026](https://tippspiel.com/en/fussball/) scoring rubric (win-exact 4, win-goal-diff 3, win-tendency 2, draw-exact 4, draw-tendency 3, wrong 0). Currently running at **~1.87 points per match** on 100 played fixtures.

## Highlights

- **Dixon–Coles bivariate Poisson** scoreline model + **XGBoost / LightGBM / LogisticRegression** ensemble for outcome, with isotonic calibration
- **Bookmaker odds blend** (median of 20+ books via the-odds-api.com), weight empirically tuned on the live tournament
- **Bayesian recalibration** against live draw / home-win rates so the model adapts as the tournament unfolds
- **Confirmed XI + referee + manager tactical matchup** features pulled from FIFA's official v3 API at T–75 min
- **Claude Sonnet 4.6** parses injury news (Google News RSS) into structured OUT / DOUBT / RETURNED status per player
- **Auto-retrains after every match** — 9-step pipeline including fresh Elo, features, models, MC simulation, odds refresh, and hyperparameter re-tune
- **KO-aware EV grid** — tips for knockout rounds never pick a draw
- **Deterministic cascade forecast** to the Final, respecting FIFA's published R32/R16/QF/SF/Final bracket
- **Telegram bot** with `/forecast`, `/upcoming`, `/bracket`, `/champion`, `/bonus`, `/refresh`, `/result`, `/predict`, `/lineup` commands

## How it works

```
history (martj42, 1872→today) ──┐
                                ├──► features (158 cols) ──► ensemble ──┐
Elo (eloratings.net method) ────┤                                        ├──► Dixon–Coles goals ──► tip EV grid
                                │                                        │
squads + injuries (Claude) ─────┤                                        │
                                │                                        │
FIFA API (schedule, lineups) ───┤                                        │
                                │                                        │
bookmaker odds (the-odds-api) ──┴───► odds blend + Bayesian recal ──────┘
```

Each played match triggers a full retrain and re-tune, so the model tracks tournament drift (draw rates, home advantage on neutral grounds, etc.) instead of freezing at pre-tournament priors.

## Sample output

`/upcoming 2` — next 48 hours grouped by date:
```
Sat 04 Oct
 M97  QF   Argentina vs Colombia    01:00
      Pick 2-1  (p_A 0.51 / p_D 0.24 / p_C 0.25)  EV 2.14 pts

 M98  QF   Spain    vs Portugal     04:30
      Pick 1-1  (p_S 0.38 / p_D 0.29 / p_P 0.33)  EV 1.89 pts
```

`/forecast` — deterministic bracket to the Final:
```
SF1  France  0-1 Spain
SF2  England 0-1 Argentina
3rd  France  1-0 England
FINAL  Spain 1-0 Argentina  🏆
```

## Quick start

Requires Python 3.11+.

```bash
git clone https://github.com/<you>/fifa-wc2026-predictor.git
cd fifa-wc2026-predictor
pip install -r requirements.txt

# 1. Copy the example env and add your keys
cp .env.example .env
# then edit .env — see "API keys" below

# 2. Build the pipeline (one-shot; ~10 minutes end-to-end)
python 01_fetch_history.py       # martj42 international results
python 02_compute_elo.py         # Elo time-series
python 15_fifa_schedule.py       # FIFA WC2026 fixtures
python 09_scrape_squads.py       # Wikipedia squads (48 teams, ~1200 players)
python 11_team_strength.py       # aggregate team profiles
python 16_injury_fetcher.py      # optional — needs ANTHROPIC_API_KEY
python 03_features.py            # 158 feature columns
python 04_train_prematch.py      # 3-class ensemble
python 08_train_goals.py         # Poisson goal rates
python 05_simulate_tournament.py # 10k Monte Carlo bracket sims
python 17_odds_fetcher.py        # optional — needs THE_ODDS_API_KEY
python 14_cascade_forecast.py    # deterministic bracket

# 3. Start the API + Telegram bot
uvicorn 06_api:app --port 8001   # or: python 06_api.py
python match_bot.py              # in a second terminal
```

On Windows, `start_api.bat` and `start_bot.bat` wrap those two commands with auto-restart.

## API keys

All optional except `ANTHROPIC_API_KEY` (if you want injury data) and Telegram credentials (if you want the bot). Add them to `.env`:

| Key                   | Purpose                                    | Where to get it |
|-----------------------|--------------------------------------------|-----------------|
| `ANTHROPIC_API_KEY`   | Structured injury extraction (Sonnet 4.6)  | https://console.anthropic.com/ |
| `FIFA_BOT_TOKEN`      | Telegram bot token                         | @BotFather on Telegram |
| `FIFA_BOT_CHAT_ID`    | Your chat ID for direct-message delivery   | @userinfobot on Telegram |
| `THE_ODDS_API_KEY`    | Bookmaker odds (500 calls/month free tier) | https://the-odds-api.com/ |
| `RAPIDAPI_KEY`        | Backup lineup source (API-Football)        | https://rapidapi.com/api-sports/api/api-football |

Tuning knobs live in `.env` too — `ODDS_BLEND`, `DC_RHO`, `DRAW_BOOST`, `CAL_BLEND`, `AUTO_TUNE_INTERVAL`. Defaults are what the auto-tuner converged to (0.80 / -0.05 / 1.00 / 0.35 / 1).

## Pipeline files

| File | Purpose |
|------|---------|
| `01_fetch_history.py`     | Pull martj42 international results (~49k matches, 1872–today) |
| `02_compute_elo.py`       | eloratings.net methodology, K=20–60, home advantage +100 |
| `03_features.py`          | 158-col feature frame (form, head-to-head, tactical, squad-derived) |
| `04_train_prematch.py`    | 3-class XGB+LGB+LR ensemble, isotonic-calibrated |
| `05_simulate_tournament.py` | 10k Monte Carlo sims with proper FIFA tiebreakers |
| `06_api.py`               | FastAPI server (port 8001) |
| `07_retrain_after_match.py` | Full 9-step retrain triggered per result |
| `08_train_goals.py`       | Poisson regression on home/away goal rates |
| `09_scrape_squads.py`     | Wikipedia squads (48 teams) |
| `11_team_strength.py`     | Aggregate 12-col team profile from squads |
| `12_referees.py`          | Referee home-win bias (capped ±5pp) |
| `13_lineup_fetcher.py`    | FIFA API → API-Football → FotMob → manual JSON |
| `14_cascade_forecast.py`  | Deterministic bracket to the Final |
| `15_fifa_schedule.py`     | Pull FIFA `/calendar/matches` |
| `16_injury_fetcher.py`    | Google News RSS + Claude Sonnet 4.6 structured extraction |
| `17_odds_fetcher.py`      | the-odds-api median across bookmakers, de-vig, merge |
| `17_auto_tune.py`         | 3D grid sweep over ODDS_BLEND × DC_RHO × DRAW_BOOST |
| `18_refresh.py`           | Data-only refresh (~75 s) — schedule + results + cascade + reload |
| `match_bot.py`            | Async Telegram bot |
| `tip_optimizer.py`        | Dixon–Coles + odds blend + KO-aware EV grid |

## License

MIT — do what you want, no warranty. See `LICENSE`.
