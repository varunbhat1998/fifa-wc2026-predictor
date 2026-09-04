"""
Fetch WC2026 injury news -> Claude Sonnet 4.6 structured extraction -> CSV.

Pipeline:
  1. Pull recent headlines from Google News RSS (aggregates BBC, Reuters, ESPN,
     Sky, Goal, The Athletic, etc.) for several injury-related queries.
  2. Deduplicate by URL/title hash. Skip anything we've already processed
     (cache file data/injuries_seen.json).
  3. Batch the unseen headlines and send to claude-sonnet-4-6 with a structured
     JSON schema: per-player record {player, team, status, severity,
     expected_return, source_title, confidence}.
  4. Append confirmed injuries (confidence >= 0.7) to data/injuries_2026.csv.
     Status: OUT / DOUBT / RETURNED.
  5. The retrain pipeline (07_retrain_after_match.py) calls this BEFORE running
     11_team_strength.py so injured players are filtered out of team profiles.

Usage:
  python 16_injury_fetcher.py            # fetch & process new headlines
  python 16_injury_fetcher.py --dry-run  # show headlines, no Claude call
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# Load .env from the project root so users can drop credentials in one place
# without setting system env vars. Silently no-ops if the file is missing.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

DATA_DIR = Path(__file__).parent / "data"
INJURIES_CSV = DATA_DIR / "injuries_2026.csv"
SEEN_CACHE = DATA_DIR / "injuries_seen.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CLAUDE_MODEL = "claude-sonnet-4-6"

# Google News RSS — public, no auth. We use multiple targeted queries so we
# don't miss country-specific coverage.
NEWS_QUERIES = [
    '"World Cup 2026" injury',
    '"World Cup 2026" ruled out',
    '"World Cup 2026" doubt',
    "World Cup squad injury 2026",
    "World Cup 2026 hamstring",
    "World Cup 2026 ACL",
]
NEWS_RSS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

# Cap the size of each Claude call to keep latency reasonable + bound cost.
MAX_HEADLINES_PER_CALL = 40

# Window of how far back to fetch. WC squads were submitted ~June 1.
LOOKBACK_DAYS = 14

# Canonical team names (must match squads_2026.csv / team_profiles_2026.csv).
WC2026_TEAMS = {
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Turkey",
    "Germany", "Curacao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Sweden", "Tunisia",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Iraq", "Norway",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama",
}


@dataclass
class Headline:
    title: str
    link: str
    pub_date: str
    source: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.link.encode("utf-8")).hexdigest()[:16]


# --------------------------- news fetching ---------------------------

def fetch_headlines() -> list[Headline]:
    """Pull recent injury-related headlines from Google News RSS for each query.
    Returns deduplicated headlines within the LOOKBACK_DAYS window."""
    from xml.etree import ElementTree as ET
    headers = {"User-Agent": "fifa-predictor/1.0 (research bot)"}
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    seen_links: set[str] = set()
    out: list[Headline] = []
    for q in NEWS_QUERIES:
        url = NEWS_RSS_URL.format(q=requests.utils.quote(q))
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
        except Exception as e:
            print(f"[news] fetch failed for query '{q}': {e}")
            continue
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            continue
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            source = (item.find("source").text if item.find("source") is not None else "").strip()
            if not (title and link) or link in seen_links:
                continue
            try:
                pub_dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %Z")
            except ValueError:
                pub_dt = datetime.utcnow()
            if pub_dt < cutoff:
                continue
            seen_links.add(link)
            out.append(Headline(title=title, link=link, pub_date=pub, source=source))
    return out


def load_seen() -> set[str]:
    if not SEEN_CACHE.exists():
        return set()
    try:
        return set(json.loads(SEEN_CACHE.read_text(encoding="utf-8")))
    except Exception:
        return set()


def save_seen(seen: set[str]) -> None:
    SEEN_CACHE.write_text(json.dumps(sorted(seen)), encoding="utf-8")


# --------------------------- Claude extraction ---------------------------

EXTRACTION_PROMPT = (
    "You are extracting structured football-injury data from news headlines about the FIFA "
    "World Cup 2026 (June 11 - July 19, 2026 in USA, Canada, Mexico). The 48 participating "
    "national teams are: " + ", ".join(sorted(WC2026_TEAMS)) + ".\n\n"
    "For each headline, decide whether it reports an injury affecting a player on one of these "
    "48 national teams. Return a JSON object {\"injuries\": [...]}. Each entry must have:\n"
    "  - \"player\":   full player name (canonical English spelling)\n"
    "  - \"team\":     one of the 48 team names exactly as listed above\n"
    "  - \"status\":   \"OUT\" (will miss tournament or current period), "
    "\"DOUBT\" (uncertain participation), or \"RETURNED\" (fit again)\n"
    "  - \"severity\": one of \"minor\", \"moderate\", \"serious\" (best guess; null if unknown)\n"
    "  - \"expected_return\": ISO date YYYY-MM-DD or null\n"
    "  - \"source_index\":  the 1-based index of the headline this came from\n"
    "  - \"source_title\":  the original headline text\n"
    "  - \"confidence\": float 0..1 — how confident you are this is a real, current, "
    "WC-relevant injury for a player on one of the 48 teams. Use <0.7 for vague rumours, "
    "old news, friendly-match knocks, or club-only injuries that don't affect WC availability.\n\n"
    "Only emit entries with confidence >= 0.5. If a headline is irrelevant (transfer news, "
    "match report, opinion, not WC-related), skip it. Return JSON only, no commentary.\n\n"
    "Headlines:\n"
)


def call_claude(headlines: list[Headline]) -> list[dict[str, Any]]:
    """Send up to MAX_HEADLINES_PER_CALL headlines to Claude, return structured records."""
    if not headlines:
        return []
    try:
        import anthropic
    except ImportError:
        print("[claude] anthropic SDK not installed. pip install anthropic", file=sys.stderr)
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[claude] ANTHROPIC_API_KEY not set — skipping LLM extraction", file=sys.stderr)
        return []
    client = anthropic.Anthropic()
    out: list[dict[str, Any]] = []
    for i in range(0, len(headlines), MAX_HEADLINES_PER_CALL):
        chunk = headlines[i : i + MAX_HEADLINES_PER_CALL]
        body = EXTRACTION_PROMPT + "\n".join(
            f"  {j + 1}. [{h.source or 'unknown'}] {h.title}"
            for j, h in enumerate(chunk)
        )
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": body}],
            )
        except Exception as e:
            print(f"[claude] call failed: {e}", file=sys.stderr)
            continue
        text = "".join(b.text for b in resp.content if b.type == "text")
        # Strip ```json fences if present.
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"```\s*$", "", text).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            print(f"[claude] non-JSON response: {text[:200]}", file=sys.stderr)
            continue
        entries = obj.get("injuries") or obj.get("results") or []
        for e in entries:
            idx = e.get("source_index")
            if isinstance(idx, int) and 1 <= idx <= len(chunk):
                e["source_link"] = chunk[idx - 1].link
            out.append(e)
        time.sleep(0.4)   # gentle pace
    return out


# --------------------------- persistence ---------------------------

def append_to_csv(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    df_new = pd.DataFrame(records)
    df_new = df_new[df_new["team"].isin(WC2026_TEAMS)]
    df_new = df_new[df_new.get("confidence", 1.0).astype(float) >= 0.7]
    if df_new.empty:
        return 0
    df_new["recorded_at"] = datetime.utcnow().isoformat()
    if INJURIES_CSV.exists():
        df_old = pd.read_csv(INJURIES_CSV, encoding="utf-8")
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    # Dedup: keep most recent record per (team, player); status may change.
    df = df.sort_values("recorded_at").drop_duplicates(
        subset=["team", "player"], keep="last"
    )
    df.to_csv(INJURIES_CSV, index=False, encoding="utf-8")
    return len(df_new)


# --------------------------- entry point ---------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Just show fetched headlines; no Claude API call.")
    args = p.parse_args()

    seen = load_seen()
    print(f"[16_injury_fetcher] cache: {len(seen)} previously seen headlines")
    all_headlines = fetch_headlines()
    print(f"[16_injury_fetcher] pulled {len(all_headlines)} headlines from Google News RSS")

    fresh = [h for h in all_headlines if h.hash not in seen]
    print(f"[16_injury_fetcher] new: {len(fresh)}")
    if args.dry_run:
        for h in fresh[:30]:
            print(f"  [{h.source}] {h.title[:140]}")
        return 0

    records = call_claude(fresh)
    n_kept = append_to_csv(records)
    print(f"[16_injury_fetcher] Claude returned {len(records)} candidates -> {n_kept} confirmed")
    if INJURIES_CSV.exists():
        df = pd.read_csv(INJURIES_CSV, encoding="utf-8")
        print(f"[16_injury_fetcher] total in {INJURIES_CSV.name}: {len(df)} rows")
        out_now = df[df["status"] == "OUT"].groupby("team").size().sort_values(ascending=False)
        if not out_now.empty:
            print("\n  current OUT count by team:")
            for t, n in out_now.items():
                print(f"    {t}: {n}")

    # Update seen cache
    seen.update(h.hash for h in fresh)
    save_seen(seen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
