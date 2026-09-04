"""
Step 12 - Aggregate squad data into per-team strength + chemistry features.

Phase A (no FBref, ships immediately):
  caps_top22          : average international caps among the top-22 most-capped
                        players (proxy for experience / depth)
  caps_top11          : top-11 caps mean (likely starters)
  goal_rate_top4_fw   : top-4 forwards' goals/cap (attack threat)
  age_top22_avg       : avg age of top-22 (maturity vs youth)
  age_spread_top22    : std of top-22 ages (peak band breadth)
  top5_league_pct     : % of players at top-5-league clubs (squad quality proxy)
  big_club_pct        : % at the world's top-30 clubs (using a hand-curated list)
  chemistry_max_cluster : largest group of squad-mates from the same club
                        (e.g., Man City supplies 4 England players)
  chemistry_top3_sum  : sum of three largest same-club clusters in the squad
  unique_clubs        : count of distinct clubs (lower = more chemistry)
  gk_avg_caps         : goalkeepers' avg caps

Phase B (later, when FBref club-form scrape lands): xG, xA, defensive actions,
mins-played weighted form 50%/30%/20% recent/12mo/career.

Output: data/team_profiles_2026.csv  one row per team.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
INJURIES_CSV = DATA_DIR / "injuries_2026.csv"


def _injured_set() -> dict[str, list[str]]:
    """Return {team: [normalised_full_name, ...]} of players currently OUT.

    Reads data/injuries_2026.csv (built by 16_injury_fetcher.py via the Claude
    extraction pipeline). Only OUT players are removed; DOUBT/RETURNED kept.
    """
    if not INJURIES_CSV.exists():
        return {}
    df = pd.read_csv(INJURIES_CSV, encoding="utf-8")
    df = df[df["status"] == "OUT"]
    out: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        nm = _normalise_full(str(r["player"]))
        if nm:
            out.setdefault(r["team"], []).append(nm)
    return out


def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )


def _normalise_full(name: str) -> str:
    """Lowercased, accent-stripped, particles dropped, hyphens preserved
    (so Korean compound surnames like 'Cho Gue-sung' stay distinct from
    'Lee Jae-sung'). Returns a single canonical string."""
    s = _strip_accents(str(name)).lower()
    # Strip parenthetical annotations like "(captain)" first.
    s = re.sub(r"\(.*?\)", " ", s)
    # Keep letters, spaces and hyphens.
    s = re.sub(r"[^a-z \-]+", " ", s)
    drop = {"jr", "sr", "ii", "iii", "da", "de", "del", "der", "di", "do",
            "dos", "la", "le", "van", "von", "el", "al", "the"}
    toks = [t for t in s.split() if t and t not in drop]
    return " ".join(toks).strip()


def _player_matches_injury(squad_name: str, injured_full: str) -> bool:
    """Tight match: the normalised injury name must be a substring of the
    normalised squad name (or equal). Single-token injuries (mononyms like
    'neymar') need length >= 5 to avoid matching short Asian-style tokens.

    Avoids false positives where two different players share one token
    (e.g. 'Florian Karl' shouldn't match 'Florian Wirtz')."""
    if not injured_full:
        return False
    squad_full = _normalise_full(squad_name)
    if not squad_full:
        return False
    if injured_full == squad_full:
        return True
    # Multi-token injury names: must appear as a substring of the squad name.
    if " " in injured_full:
        return injured_full in squad_full
    # Single-token (mononym) injuries: substring OK only when distinctive.
    if len(injured_full) >= 5:
        # Either a whole-token match (split by space/hyphen) OR substring inside
        # a single squad token.
        squad_tokens = re.split(r"[ \-]+", squad_full)
        if injured_full in squad_tokens:
            return True
        for tok in squad_tokens:
            if injured_full in tok and len(injured_full) >= len(tok) - 2:
                return True
    return False


import re  # noqa: E402

# Top-5 European leagues' major clubs (loose: clubs that finished in top-half
# of the Big 5 leagues in 2024-25, plus elite Saudi/MLS clubs that signed stars).
# Used for top5_league_pct + big_club_pct features. List is intentionally broad
# rather than exhaustive — feature is a percentage so a few omissions are fine.
BIG5_LEAGUE_CLUBS = {
    # Premier League
    "Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United",
    "Tottenham Hotspur", "Newcastle United", "Aston Villa", "Brighton & Hove Albion",
    "West Ham United", "Crystal Palace", "Fulham", "Brentford", "Wolverhampton Wanderers",
    "Everton", "Nottingham Forest", "Bournemouth", "Sheffield United", "Burnley",
    "Leicester City", "Ipswich Town", "Southampton", "Leeds United",
    # La Liga
    "Real Madrid", "Barcelona", "Atlético Madrid", "Atletico Madrid", "Athletic Bilbao",
    "Real Sociedad", "Real Betis", "Sevilla", "Villarreal", "Valencia",
    "Girona", "Mallorca", "Las Palmas", "Osasuna", "Alavés", "Alaves",
    "Getafe", "Celta Vigo", "Rayo Vallecano", "Espanyol",
    # Bundesliga
    "Bayern Munich", "Bayer Leverkusen", "RB Leipzig", "Borussia Dortmund",
    "VfB Stuttgart", "Eintracht Frankfurt", "TSG Hoffenheim", "Hoffenheim",
    "Borussia Mönchengladbach", "Mönchengladbach", "VfL Wolfsburg", "Wolfsburg",
    "SC Freiburg", "Freiburg", "FSV Mainz 05", "Mainz 05", "Union Berlin",
    "Werder Bremen", "FC Augsburg", "Augsburg", "1. FC Heidenheim", "Heidenheim",
    "VfL Bochum", "Bochum", "Holstein Kiel", "St. Pauli", "FC St. Pauli",
    "Hamburger SV",
    # Serie A
    "Inter Milan", "Internazionale", "AC Milan", "Juventus", "Napoli", "Roma",
    "Atalanta", "Lazio", "Bologna", "Fiorentina", "Torino", "Hellas Verona",
    "Como", "Cagliari", "Empoli", "Genoa", "Lecce", "Monza", "Parma", "Udinese",
    "Sassuolo", "Venezia",
    # Ligue 1
    "Paris Saint-Germain", "Monaco", "AS Monaco", "Marseille", "Olympique Marseille",
    "Lille", "OSC Lille", "Lyon", "Olympique Lyonnais", "Nice", "OGC Nice",
    "Lens", "RC Lens", "Strasbourg", "RC Strasbourg", "Rennes", "Stade Rennais",
    "Reims", "Nantes", "FC Nantes", "Toulouse", "Toulouse FC", "Montpellier",
    "Brest", "Stade Brestois", "Saint-Étienne", "Saint-Etienne", "Angers", "Le Havre",
    "Auxerre",
}

# Elite "big-club" tier — used for `big_club_pct`. Hand-picked top-30 from
# 2024-25 UEFA/Conmebol club Elo + Saudi big-money signings.
BIG_CLUBS = {
    "Manchester City", "Real Madrid", "Bayern Munich", "Paris Saint-Germain",
    "Liverpool", "Arsenal", "Barcelona", "Inter Milan", "Internazionale",
    "Atlético Madrid", "Atletico Madrid", "Bayer Leverkusen", "Borussia Dortmund",
    "Chelsea", "Manchester United", "Tottenham Hotspur", "Napoli", "Juventus",
    "AC Milan", "Roma", "Atalanta", "Newcastle United", "Aston Villa",
    "Real Sociedad", "Athletic Bilbao", "RB Leipzig", "VfB Stuttgart",
    "Eintracht Frankfurt", "Marseille", "Olympique Marseille", "Lille", "OSC Lille",
    "Brighton & Hove Albion", "Sporting CP", "Benfica", "Porto", "FC Porto",
    "Ajax", "PSV Eindhoven", "Feyenoord", "Galatasaray", "Fenerbahçe", "Fenerbahce",
    # Saudi big-money
    "Al-Hilal", "Al Hilal", "Al-Nassr", "Al Nassr", "Al-Ittihad", "Al Ittihad",
    "Al-Ahli", "Al Ahli",
}


def _pos_filter(squad: pd.DataFrame, pos: str) -> pd.DataFrame:
    return squad[squad["position"].str.contains(pos, case=False, na=False)]


def _largest_club_clusters(squad: pd.DataFrame, k: int = 3) -> tuple[int, int, int]:
    """Return (largest cluster, sum of top-k clusters, unique club count)."""
    counts = Counter(squad["club"].dropna())
    sizes = sorted(counts.values(), reverse=True)
    largest = sizes[0] if sizes else 0
    top_k_sum = sum(sizes[:k])
    return largest, top_k_sum, len(counts)


def compute_team_profile(squad: pd.DataFrame) -> dict:
    squad = squad.copy()
    squad["caps"] = squad["caps"].fillna(0).astype(int)
    squad["goals"] = squad["goals"].fillna(0).astype(int)
    squad["age"] = squad["age"].fillna(squad["age"].median()).astype(float)

    # Rank by caps; top-N approximates likely-XI ordering (most-capped picked).
    top22 = squad.sort_values("caps", ascending=False).head(22)
    top11 = squad.sort_values("caps", ascending=False).head(11)
    fws = _pos_filter(squad, "FW").sort_values("goals", ascending=False).head(4)
    gks = _pos_filter(squad, "GK")

    largest, top3_sum, n_clubs = _largest_club_clusters(squad, k=3)

    cap_total = squad["caps"].sum() or 1
    top5_pct = squad["club"].isin(BIG5_LEAGUE_CLUBS).mean()
    big_club_pct = squad["club"].isin(BIG_CLUBS).mean()

    return {
        "caps_top22": float(top22["caps"].mean()),
        "caps_top11": float(top11["caps"].mean()),
        "goal_rate_top4_fw": float(
            fws["goals"].sum() / max(fws["caps"].sum(), 1)
        ),
        "fw_goals_top4_total": int(fws["goals"].sum()),
        "age_top22_avg": float(top22["age"].mean()),
        "age_top22_std": float(top22["age"].std() or 0.0),
        "top5_league_pct": float(top5_pct),
        "big_club_pct": float(big_club_pct),
        "chemistry_max_cluster": int(largest),
        "chemistry_top3_sum": int(top3_sum),
        "unique_clubs": int(n_clubs),
        "gk_avg_caps": float(gks["caps"].mean()) if len(gks) else 0.0,
        "squad_size": int(len(squad)),
    }


def main() -> int:
    sq = pd.read_csv(DATA_DIR / "squads_2026.csv")
    injured = _injured_set()
    profiles: list[dict] = []
    n_applied_total = 0
    n_teams_affected = 0
    for team, sub in sq.groupby("team"):
        out_full_names = injured.get(team, [])
        n_total = len(sub)
        n_matched_here = 0
        if out_full_names:
            keep = []
            removed_names = []
            for _, r in sub.iterrows():
                player_name = str(r["player"])
                hit = False
                for inj_full in out_full_names:
                    if _player_matches_injury(player_name, inj_full):
                        hit = True
                        break
                if hit:
                    n_matched_here += 1
                    removed_names.append(player_name)
                    continue
                keep.append(r)
            if removed_names:
                try:
                    print(f"  [{team}] OUT: " + ", ".join(removed_names))
                except UnicodeEncodeError:
                    print(f"  [{team}] OUT: " + ", ".join(removed_names).encode("ascii", "replace").decode("ascii"))
            if keep:
                sub = pd.DataFrame(keep)
        n_applied_total += n_matched_here
        if n_matched_here:
            n_teams_affected += 1
        prof = compute_team_profile(sub)
        prof["team"] = team
        prof["n_injured_out"] = n_total - len(sub)
        prof["availability_pct"] = len(sub) / max(n_total, 1)
        profiles.append(prof)
    if injured:
        print(f"[11_team_strength] applied {n_applied_total} OUT players "
              f"across {n_teams_affected} teams (from {INJURIES_CSV.name})")
    out = pd.DataFrame(profiles).sort_values("caps_top11", ascending=False).reset_index(drop=True)
    out.to_csv(DATA_DIR / "team_profiles_2026.csv", index=False)
    print(f"[11_team_strength] wrote {DATA_DIR / 'team_profiles_2026.csv'}")
    cols = [
        "team", "caps_top11", "caps_top22", "age_top22_avg",
        "top5_league_pct", "big_club_pct",
        "chemistry_max_cluster", "chemistry_top3_sum", "fw_goals_top4_total",
    ]
    print("\nTop 10 by caps_top11:")
    print(out[cols].head(10).to_string(index=False))
    print("\nBottom 10:")
    print(out[cols].tail(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
