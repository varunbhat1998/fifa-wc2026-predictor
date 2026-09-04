"""
Hard-coded WC 2026 fixture list (groups + knockout bracket).

72 group matches (June 11-27, 2026) — confirmed from ESPN's published schedule.
32 knockout matches (June 28 - July 19, 2026) — slot-based; teams resolved by
the Monte Carlo simulator using group standings.

Team-name normalisation maps ESPN/FIFA spellings (Türkiye, Czechia) to the
martj42 canonical labels (Turkey, Czech Republic) that our Elo + model use.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# ESPN / FIFA  ->  martj42 canonical
TEAM_NAME_MAP = {
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "United States": "United States",
    "South Korea": "South Korea",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Curaçao": "Curacao",
    "Curacao": "Curacao",
    "Cape Verde": "Cape Verde",
    "DR Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
}


def canonical(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


# ---- GROUPS (final draw, 5 December 2025) ----

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


@dataclass
class Fixture:
    date: datetime
    home: str
    away: str
    city: str
    group: str | None
    stage: str          # "group" | "R32" | "R16" | "QF" | "SF" | "3rd" | "Final"
    neutral: bool       # True unless the home country == host country and home team plays at home
    match_no: int       # 1..104 in schedule order
    # FIFA primary keys, populated when this fixture was loaded from the FIFA
    # schedule CSV. Both empty strings if loaded from the hardcoded fallback.
    id_match: str = ""
    id_stage: str = ""
    referee: str | None = None
    referee_country: str | None = None
    stadium_id: str = ""
    stadium_name: str = ""


# ---- GROUP STAGE (72 matches, ESPN-confirmed) ----
# Format: (date, home, away, city, group)
_GROUP_RAW: list[tuple[str, str, str, str, str]] = [
    # Matchday 1
    ("2026-06-11", "Mexico", "South Africa", "Mexico City", "A"),
    ("2026-06-11", "South Korea", "Czechia", "Zapopan", "A"),
    ("2026-06-12", "Canada", "Bosnia and Herzegovina", "Toronto", "B"),
    ("2026-06-12", "United States", "Paraguay", "Inglewood", "D"),
    ("2026-06-13", "Qatar", "Switzerland", "Santa Clara", "B"),
    ("2026-06-13", "Brazil", "Morocco", "East Rutherford", "C"),
    ("2026-06-13", "Haiti", "Scotland", "Foxborough", "C"),
    ("2026-06-13", "Australia", "Türkiye", "Vancouver", "D"),
    ("2026-06-14", "Germany", "Curaçao", "Houston", "E"),
    ("2026-06-14", "Netherlands", "Japan", "Arlington", "F"),
    ("2026-06-14", "Ivory Coast", "Ecuador", "Philadelphia", "E"),
    ("2026-06-14", "Sweden", "Tunisia", "Guadalupe", "F"),
    ("2026-06-15", "Spain", "Cape Verde", "Atlanta", "H"),
    ("2026-06-15", "Belgium", "Egypt", "Seattle", "G"),
    ("2026-06-15", "Saudi Arabia", "Uruguay", "Miami Gardens", "H"),
    ("2026-06-15", "Iran", "New Zealand", "Inglewood", "G"),
    ("2026-06-16", "France", "Senegal", "East Rutherford", "I"),
    ("2026-06-16", "Iraq", "Norway", "Foxborough", "I"),
    ("2026-06-16", "Argentina", "Algeria", "Kansas City", "J"),
    ("2026-06-16", "Austria", "Jordan", "Santa Clara", "J"),
    ("2026-06-17", "Portugal", "DR Congo", "Houston", "K"),
    ("2026-06-17", "England", "Croatia", "Arlington", "L"),
    ("2026-06-17", "Ghana", "Panama", "Toronto", "L"),
    ("2026-06-17", "Uzbekistan", "Colombia", "Mexico City", "K"),
    # Matchday 2
    ("2026-06-18", "Czechia", "South Africa", "Atlanta", "A"),
    ("2026-06-18", "Switzerland", "Bosnia and Herzegovina", "Inglewood", "B"),
    ("2026-06-18", "Canada", "Qatar", "Vancouver", "B"),
    ("2026-06-18", "Mexico", "South Korea", "Zapopan", "A"),
    ("2026-06-19", "United States", "Australia", "Seattle", "D"),
    ("2026-06-19", "Scotland", "Morocco", "Foxborough", "C"),
    ("2026-06-19", "Brazil", "Haiti", "Philadelphia", "C"),
    ("2026-06-19", "Türkiye", "Paraguay", "Santa Clara", "D"),
    ("2026-06-20", "Netherlands", "Sweden", "Houston", "F"),
    ("2026-06-20", "Germany", "Ivory Coast", "Toronto", "E"),
    ("2026-06-20", "Ecuador", "Curaçao", "Kansas City", "E"),
    ("2026-06-20", "Tunisia", "Japan", "Guadalupe", "F"),
    ("2026-06-21", "Spain", "Saudi Arabia", "Atlanta", "H"),
    ("2026-06-21", "Belgium", "Iran", "Inglewood", "G"),
    ("2026-06-21", "Uruguay", "Cape Verde", "Miami Gardens", "H"),
    ("2026-06-21", "New Zealand", "Egypt", "Vancouver", "G"),
    ("2026-06-22", "Argentina", "Austria", "Arlington", "J"),
    ("2026-06-22", "France", "Iraq", "Philadelphia", "I"),
    ("2026-06-22", "Norway", "Senegal", "East Rutherford", "I"),
    ("2026-06-22", "Jordan", "Algeria", "Santa Clara", "J"),
    ("2026-06-23", "Portugal", "Uzbekistan", "Houston", "K"),
    ("2026-06-23", "England", "Ghana", "Foxborough", "L"),
    ("2026-06-23", "Panama", "Croatia", "Toronto", "L"),
    ("2026-06-23", "Colombia", "DR Congo", "Zapopan", "K"),
    # Matchday 3
    ("2026-06-24", "Switzerland", "Canada", "Vancouver", "B"),
    ("2026-06-24", "Bosnia and Herzegovina", "Qatar", "Seattle", "B"),
    ("2026-06-24", "Scotland", "Brazil", "Miami Gardens", "C"),
    ("2026-06-24", "Morocco", "Haiti", "Atlanta", "C"),
    ("2026-06-24", "Czechia", "Mexico", "Mexico City", "A"),
    ("2026-06-24", "South Africa", "South Korea", "Guadalupe", "A"),
    ("2026-06-25", "Ecuador", "Germany", "East Rutherford", "E"),
    ("2026-06-25", "Curaçao", "Ivory Coast", "Philadelphia", "E"),
    ("2026-06-25", "Japan", "Sweden", "Arlington", "F"),
    ("2026-06-25", "Tunisia", "Netherlands", "Kansas City", "F"),
    ("2026-06-25", "Türkiye", "United States", "Inglewood", "D"),
    ("2026-06-25", "Paraguay", "Australia", "Santa Clara", "D"),
    ("2026-06-26", "Norway", "France", "Foxborough", "I"),
    ("2026-06-26", "Senegal", "Iraq", "Toronto", "I"),
    ("2026-06-26", "Cape Verde", "Saudi Arabia", "Houston", "H"),
    ("2026-06-26", "Uruguay", "Spain", "Zapopan", "H"),
    ("2026-06-26", "Egypt", "Iran", "Seattle", "G"),
    ("2026-06-26", "New Zealand", "Belgium", "Vancouver", "G"),
    ("2026-06-27", "Panama", "England", "East Rutherford", "L"),
    ("2026-06-27", "Croatia", "Ghana", "Philadelphia", "L"),
    ("2026-06-27", "Colombia", "Portugal", "Miami Gardens", "K"),
    ("2026-06-27", "DR Congo", "Uzbekistan", "Atlanta", "K"),
    ("2026-06-27", "Algeria", "Austria", "Kansas City", "J"),
    ("2026-06-27", "Jordan", "Argentina", "Arlington", "J"),
]

HOST_COUNTRIES = {"Mexico", "United States", "Canada"}
MEXICAN_CITIES = {"Mexico City", "Zapopan", "Guadalajara", "Monterrey", "Guadalupe"}
CANADIAN_CITIES = {"Toronto", "Vancouver"}

def _host_country(city: str) -> str:
    if city in MEXICAN_CITIES:
        return "Mexico"
    if city in CANADIAN_CITIES:
        return "Canada"
    return "United States"


def build_group_fixtures() -> list[Fixture]:
    out: list[Fixture] = []
    for i, (d, h, a, city, grp) in enumerate(_GROUP_RAW, start=1):
        home = canonical(h)
        away = canonical(a)
        host = _host_country(city)
        # Match is "home" only if the listed home team's nation matches the host country.
        # Everything else is neutral for Elo / feature purposes.
        is_neutral = home != host
        out.append(Fixture(
            date=datetime.strptime(d, "%Y-%m-%d"),
            home=home,
            away=away,
            city=city,
            group=grp,
            stage="group",
            neutral=is_neutral,
            match_no=i,
        ))
    return out


# ---- KNOCKOUT BRACKET (32 matches, slot-based) ----
# These are placeholder slots resolved by the Monte Carlo simulator. The bracket
# pairing is the actual published 2026 FIFA bracket — winners flow:
#   R32 (16 matches) -> R16 (8) -> QF (4) -> SF (2) -> Final (1) + 3rd place play-off
KNOCKOUT_DATES = {
    "R32":   ("2026-06-28", "2026-07-03"),
    "R16":   ("2026-07-04", "2026-07-07"),
    "QF":    ("2026-07-09", "2026-07-11"),
    "SF":    ("2026-07-14", "2026-07-15"),
    "3rd":   ("2026-07-18", "2026-07-18"),
    "Final": ("2026-07-19", "2026-07-19"),
}


def all_group_fixtures() -> list[Fixture]:
    """Return all 72 group-stage fixtures, preferring the FIFA-API-pulled CSV
    when it exists (data/wc2026_fifa_schedule.csv) and falling back to the
    hardcoded ESPN-derived list otherwise."""
    fifa = _load_fifa_fixtures()
    if fifa:
        group = [f for f in fifa if f.stage == "group"]
        if len(group) >= 60:   # tolerate a couple of missing rows
            return group
    return build_group_fixtures()


def all_fixtures_including_ko() -> list[Fixture]:
    """All 104 matches when loaded from FIFA (group + R32 + R16 + QF + SF + 3rd + Final).
    Returns only group matches if the FIFA CSV is missing."""
    fifa = _load_fifa_fixtures()
    return fifa if fifa else build_group_fixtures()


def _load_fifa_fixtures() -> list[Fixture] | None:
    import pandas as pd
    from datetime import datetime as _dt
    from pathlib import Path as _Path
    path = _Path(__file__).parent / "data" / "wc2026_fifa_schedule.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return None
    # Map FIFA stadium_name -> our cities for altitude lookup compatibility.
    # FIFA's "Mexico City Stadium", "Guadalajara Stadium" etc. lose the venue
    # specifics, but we mainly use these for the altitude flag, which only
    # depends on the host city — so the mapping is straightforward.
    STADIUM_CITY = {
        "Mexico City Stadium":       "Mexico City",
        "Guadalajara Stadium":       "Zapopan",         # Estadio Akron
        "Monterrey Stadium":         "Guadalupe",       # Estadio BBVA
        "Toronto Stadium":           "Toronto",
        "Vancouver Stadium":         "Vancouver",
        "Atlanta Stadium":           "Atlanta",
        "Boston Stadium":            "Foxborough",
        "Dallas Stadium":            "Arlington",
        "Houston Stadium":           "Houston",
        "Kansas City Stadium":       "Kansas City",
        "Los Angeles Stadium":       "Inglewood",
        "Miami Stadium":             "Miami Gardens",
        "New York/New Jersey Stadium": "East Rutherford",
        "Philadelphia Stadium":      "Philadelphia",
        "San Francisco Bay Area Stadium": "Santa Clara",
        "Seattle Stadium":           "Seattle",
    }
    def _s(v) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return ""
        return str(v).strip()

    out: list[Fixture] = []
    for _, r in df.iterrows():
        sn = _s(r.get("stadium_name"))
        city = STADIUM_CITY.get(sn, sn)
        host = _host_country(city)
        home = _s(r.get("home"))
        away = _s(r.get("away"))
        if not home or not away:
            # Knockout slot whose teams aren't decided yet — keep with placeholder.
            home = home or f"TBD_{r['stage']}_{int(r['match_no'])}_H"
            away = away or f"TBD_{r['stage']}_{int(r['match_no'])}_A"
        neutral = home not in HOST_COUNTRIES or host != home
        try:
            date = _dt.fromisoformat(str(r["date_utc"]).replace("Z", "+00:00"))
        except Exception:
            continue
        out.append(Fixture(
            date=date.replace(tzinfo=None),
            home=home, away=away,
            city=city,
            group=_s(r.get("group_label")) or None,
            stage=_s(r.get("stage")) or "group",
            neutral=neutral,
            match_no=int(r["match_no"]),
            id_match=_s(r.get("id_match")),
            id_stage=_s(r.get("id_stage")),
            referee=(_s(r.get("referee")) or None),
            referee_country=(_s(r.get("referee_country")) or None),
            stadium_id=_s(r.get("stadium_id")),
            stadium_name=sn,
        ))
    return out


if __name__ == "__main__":
    fxts = all_group_fixtures()
    print(f"WC2026 group-stage fixtures: {len(fxts)}")
    print(f"first: {fxts[0]}")
    print(f"last : {fxts[-1]}")
    teams_in_groups = sorted({t for grp in GROUPS.values() for t in grp})
    print(f"teams in groups: {len(teams_in_groups)}")
    teams_in_fxts = sorted({f.home for f in fxts} | {f.away for f in fxts})
    print(f"teams in fixtures: {len(teams_in_fxts)}")
    missing = set(teams_in_groups) - set(teams_in_fxts)
    extra = set(teams_in_fxts) - set(teams_in_groups)
    if missing or extra:
        print(f"  MISMATCH! missing from fixtures: {missing}")
        print(f"  not in groups: {extra}")
