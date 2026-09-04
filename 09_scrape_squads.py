"""
Step 11 - Scrape the 2026 FIFA World Cup squads from Wikipedia.

Single source: https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads
All 48 squads (~1,248 players) are on one page in HTML tables of the form:
  No. | Pos. | Player | DOB (age) | Caps | Goals | Club

Output: data/squads_2026.csv  with columns
  team, jersey_no, position, player, dob, age, caps, goals, club, club_country
"""
from __future__ import annotations

import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent / "data"
URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

# Map every team-header form Wikipedia uses to our canonical labels.
HEADING_MAP = {
    "United States": "United States", "USA": "United States",
    "Türkiye": "Turkey", "Turkiye": "Turkey",
    "Czechia": "Czech Republic", "Curaçao": "Curacao",
    "Côte d'Ivoire": "Ivory Coast", "DR Congo": "DR Congo",
    "Cape Verde": "Cape Verde",
}


def canonical(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"\s*\[.*?\]\s*$", "", name)  # strip footnote refs
    return HEADING_MAP.get(name, name)


def parse_age(dob_cell: str) -> tuple[str, int | None]:
    """Wikipedia DOB cell often looks like '(1995-04-12) 12 April 1995 (aged 31)'.
    Extract the (YYYY-MM-DD) and age."""
    if not isinstance(dob_cell, str):
        return "", None
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", dob_cell)
    dob = f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else ""
    age_m = re.search(r"aged\s+(\d+)", dob_cell)
    age = int(age_m.group(1)) if age_m else None
    return dob, age


def parse_int(cell) -> int | None:
    if cell is None or (isinstance(cell, float) and cell != cell):
        return None
    s = str(cell).strip()
    s = re.sub(r"\[.*?\]", "", s)        # footnote refs
    s = re.sub(r"[^\d\-]", "", s)        # commas, spaces
    return int(s) if s and s.lstrip("-").isdigit() else None


def parse_club_cell(cell) -> tuple[str, str]:
    """The Club column is usually '<flag> Club Name' — flag implies country."""
    if not isinstance(cell, str):
        return "", ""
    s = re.sub(r"\[.*?\]", "", cell).strip()
    parts = s.split(" ", 1)
    # Wikipedia 'data-sort-value' usually drops the flag; pandas keeps the text only.
    return s, ""


def fetch_html() -> str:
    print(f"[09_scrape_squads] fetching {URL} ...")
    r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (fifa-predictor)"}, timeout=30)
    r.raise_for_status()
    return r.text


def parse_squads(html: str) -> pd.DataFrame:
    """Walk a flat list of headings + tables. Each H3 is a team name; the next
    sibling table is that team's squad. H2 may carry a Group letter."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    current_team: str | None = None
    current_group: str | None = None

    SKIP_HEADINGS = {
        "Contents", "Statistics", "Goalscorers", "Notes", "See also",
        "References", "External links", "Most appearances",
        "Most goals", "Squads", "Player representation by league",
        "Player representation by club",
    }
    elems = soup.find_all(["h2", "h3", "h4", "table"])
    for tag in elems:
        name = tag.name
        if name in {"h2", "h3", "h4"}:
            txt = tag.get_text(" ", strip=True)
            txt_clean = re.sub(r"\[edit\]\s*$", "", txt).strip()
            gm = re.match(r"^Group\s+([A-L])\b", txt_clean)
            if gm:
                current_group = gm.group(1)
                current_team = None
                continue
            if txt_clean in SKIP_HEADINGS:
                current_team = None
                continue
            if name == "h3":
                current_team = canonical(txt_clean)
            continue
        if name != "table":
            continue
        if not current_team:
            continue
        if "wikitable" not in (tag.get("class") or []):
            continue
        try:
            tdf_list = pd.read_html(StringIO(str(tag)))
        except ValueError:
            continue
        for tdf in tdf_list:
            if tdf.shape[1] < 6:
                continue
            cols = [str(c).strip().lower() for c in tdf.columns]
            if not any("pos" in c for c in cols):
                continue
            if not any("player" in c or "name" in c for c in cols):
                continue
            def find(*pats):
                for i, c in enumerate(cols):
                    for p in pats:
                        if p in c:
                            return i
                return None
            ic_no = find("no.", "no ", "number")
            ic_pos = find("pos")
            ic_player = find("player", "name")
            ic_dob = find("date of birth", "dob", "birth")
            ic_caps = find("caps")
            ic_goals = find("goals")
            ic_club = find("club")
            for _, r in tdf.iterrows():
                player = r.iloc[ic_player] if ic_player is not None else None
                if pd.isna(player) or not str(player).strip():
                    continue
                dob_raw = r.iloc[ic_dob] if ic_dob is not None else ""
                dob, age = parse_age(str(dob_raw))
                club_raw = r.iloc[ic_club] if ic_club is not None else ""
                club, _ = parse_club_cell(str(club_raw))
                rows.append({
                    "team": current_team,
                    "group": current_group,
                    "jersey_no": parse_int(r.iloc[ic_no]) if ic_no is not None else None,
                    "position": str(r.iloc[ic_pos]).strip() if ic_pos is not None else "",
                    "player": str(player).strip(),
                    "dob": dob,
                    "age": age,
                    "caps": parse_int(r.iloc[ic_caps]) if ic_caps is not None else None,
                    "goals": parse_int(r.iloc[ic_goals]) if ic_goals is not None else None,
                    "club": club,
                })
            break  # only the first matching table per team
        current_team = None  # consume this team's table
    return pd.DataFrame(rows)


def main() -> int:
    html = fetch_html()
    df = parse_squads(html)
    # Drop any spurious teams not in our schedule (e.g., note tables).
    import wc2026_schedule as ws
    valid = {t for ts in ws.GROUPS.values() for t in ts}
    df = df[df["team"].isin(valid)].copy()
    out = DATA_DIR / "squads_2026.csv"
    df.to_csv(out, index=False)
    print(f"[09_scrape_squads] wrote {out}")
    print(f"  players       : {len(df):,}")
    print(f"  teams covered : {df['team'].nunique()}/48")
    per_team = df.groupby("team").size()
    too_small = per_team[per_team < 23]
    if not too_small.empty:
        print(f"  WARNING — these teams have <23 players parsed:")
        for t, n in too_small.items():
            print(f"    {t}: {n}")
    print(f"\n  sample:")
    print(df.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
