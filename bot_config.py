"""Bot configuration. Env-driven so secrets stay out of the repo."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).parent

# Load `.env` from the project root if present (python-dotenv).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
CHECKPOINT_DIR = DATA_DIR / "bot_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Telegram ----
TELEGRAM_TOKEN = os.environ.get("FIFA_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("FIFA_BOT_CHAT_ID", "")
TELEGRAM_API = "https://api.telegram.org"

# ---- Prediction API ----
API_BASE = os.environ.get("FIFA_API_BASE", "http://127.0.0.1:8001")

# ---- Cadences (seconds) ----
# Initial squad-only prediction goes out at T-PREDICT_LEAD_MIN.
# Then we poll for confirmed XI; once found, a refined prediction is sent.
PREDICT_LEAD_MIN = 180        # post initial squad-based prediction 3h before kickoff
LINEUP_POLL_START_MIN = 90    # start looking for confirmed XI at T-90min
LINEUP_POLL_INTERVAL_SEC = 90 # how often to retry until found
LINEUP_FINAL_REFRESH_MIN = 15 # last-chance refresh point (close to kickoff)
SCHEDULER_TICK_SEC = 30       # main loop tick
RESULT_POLL_SEC = 1800        # 30 min between result polls (martj42 is daily-ish)
TELEGRAM_POLL_SEC = 3         # long-poll cadence for commands

# Default kickoff time when the schedule only has a date (most matches kick off
# between 12:00 and 21:00 local; we use a single approximation).
DEFAULT_KICKOFF_HOUR_UTC = 19

# ---- External data source for result polling ----
MARTJ42_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

# ---- Retrain ----
RESULTS_ACTUAL_CSV = DATA_DIR / "results_actual.csv"
