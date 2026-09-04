"""Per-fixture state machine + checkpoint persistence."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from bot_config import CHECKPOINT_DIR


class Phase(str, Enum):
    SCHEDULED = "SCHEDULED"
    PREDICTED = "PREDICTED"            # initial (squad-based) prediction sent
    LINEUP_POLLING = "LINEUP_POLLING"  # actively looking for confirmed XI
    LINEUP_REFINED = "LINEUP_REFINED"  # confirmed-XI prediction sent
    AWAITING_RESULT = "AWAITING_RESULT"
    RESULT_IN = "RESULT_IN"
    RETRAINED = "RETRAINED"
    DONE = "DONE"


@dataclass
class MatchState:
    match_no: int
    home: str
    away: str
    city: str
    group: str | None
    stage: str
    neutral: bool
    kickoff: datetime
    phase: Phase = Phase.SCHEDULED
    prediction: dict | None = None     # p_home, p_draw, p_away, ...
    final_home_score: int | None = None
    final_away_score: int | None = None
    last_action_at: datetime = field(default_factory=datetime.utcnow)
    # FIFA primary keys + pre-assigned officials. Empty / None when loaded from
    # the hardcoded fallback schedule.
    id_match: str = ""
    id_stage: str = ""
    referee: str | None = None
    referee_country: str | None = None
    stadium_name: str = ""

    def path(self) -> Path:
        return CHECKPOINT_DIR / f"match_{self.match_no:03d}.pkl"

    def save(self) -> None:
        self.last_action_at = datetime.utcnow()
        with open(self.path(), "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, match_no: int) -> "MatchState | None":
        p = CHECKPOINT_DIR / f"match_{match_no:03d}.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kickoff"] = self.kickoff.isoformat()
        d["last_action_at"] = self.last_action_at.isoformat()
        d["phase"] = self.phase.value
        return d


def all_checkpoints() -> list[MatchState]:
    out = []
    for p in sorted(CHECKPOINT_DIR.glob("match_*.pkl")):
        try:
            with open(p, "rb") as f:
                out.append(pickle.load(f))
        except Exception:
            continue
    return out
