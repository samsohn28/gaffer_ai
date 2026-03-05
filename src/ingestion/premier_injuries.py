"""
Parse premierinjuries.com email notifications and save to data/raw/injuries.json.

Connects to Gmail via IMAP, searches for emails from premierinjuries.com,
and parses injury tables from the HTML body.

Requires in .env:
    GMAIL_ADDRESS      — your Gmail address
    GMAIL_APP_PASSWORD — Gmail App Password (not your main password)

Usage:
    python -m src.ingestion.premier_injuries
"""

import email
import imaplib
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "bronze"
OUTPUT_PATH = RAW_DIR / "injuries.json"
ELEMENTS_PATH = RAW_DIR / "elements.json"

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
SENDER_FILTER = "premierinjuries.com"


# ---------------------------------------------------------------------------
# Gmail helpers
# ---------------------------------------------------------------------------

def _load_credentials() -> tuple[str, str]:
    """Load Gmail credentials from .env; exit with a clear error if missing."""
    load_dotenv()
    import os
    address = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")
    missing = [name for name, val in [("GMAIL_ADDRESS", address), ("GMAIL_APP_PASSWORD", password)] if not val]
    if missing:
        print(
            f"Error: missing environment variable(s): {', '.join(missing)}\n"
            "Add them to your .env file:\n"
            "  GMAIL_ADDRESS=you@gmail.com\n"
            "  GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx",
            file=sys.stderr,
        )
        sys.exit(1)
    return address, password  # type: ignore[return-value]


def fetch_emails(n: int = 10) -> list[str]:
    """
    Connect to Gmail via IMAP and return HTML bodies of the n most recent
    emails from premierinjuries.com.
    """
    address, password = _load_credentials()

    mail = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try:
        mail.login(address, password)
    except imaplib.IMAP4.error as e:
        print(f"Error: Gmail login failed: {e}", file=sys.stderr)
        sys.exit(1)

    mail.select("inbox")
    status, data = mail.search(None, f'FROM "{SENDER_FILTER}"')
    if status != "OK":
        print("Error: IMAP SEARCH failed.", file=sys.stderr)
        sys.exit(1)

    msg_ids = data[0].split()
    if not msg_ids:
        print(f"  Warning: 0 emails found from '{SENDER_FILTER}'.", file=sys.stderr)
        mail.logout()
        return []

    # Take the n most recent
    recent_ids = msg_ids[-n:][::-1]
    bodies: list[str] = []

    for msg_id in recent_ids:
        status, msg_data = mail.fetch(msg_id, "(RFC822)")
        if status != "OK":
            continue
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        html = _extract_html(msg)
        if html:
            bodies.append(html)

    mail.logout()
    return bodies


def _extract_html(msg: email.message.Message) -> str | None:
    """Return the HTML part of a MIME email, or None."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
    else:
        if msg.get_content_type() == "text/html":
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_email_body(html: str) -> list[dict]:
    """
    Parse an injury table from a premierinjuries.com email HTML body.

    Stub — the actual parser will be refined once the first real email arrives.
    Currently attempts a BeautifulSoup table parse with a regex fallback.
    Returns [] with a warning if no structure is found.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print(
            "Error: beautifulsoup4 is required. Run: pip install beautifulsoup4 lxml",
            file=sys.stderr,
        )
        sys.exit(1)

    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        if not any(h in headers for h in ("player", "status", "reason")):
            continue

        for tr in rows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < len(headers):
                continue

            row = dict(zip(headers, cells))
            status_raw = row.get("status", "").strip()
            confidence = 0 if status_raw.lower() == "ruled out" else _parse_confidence(status_raw)

            records.append({
                "player_name": row.get("player", ""),
                "team": row.get("team", ""),
                "reason": row.get("reason", ""),
                "further_detail": row.get("further detail", row.get("detail", "")),
                "potential_return": row.get("potential return", ""),
                "condition": row.get("condition", ""),
                "confidence": confidence,
            })

        if records:
            return records

    # Fallback: regex scan for confidence percentages
    pattern = re.compile(r"([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s+(\d{1,3})%")
    for m in pattern.finditer(html):
        records.append({
            "player_name": m.group(1),
            "team": "",
            "reason": "",
            "further_detail": "",
            "potential_return": "",
            "condition": "",
            "confidence": min(int(m.group(2)), 100),
        })

    if not records:
        warnings.warn("parse_email_body: no injury structure found in email body.")

    return records


def _parse_confidence(text: str) -> int:
    digits = re.sub(r"[^\d]", "", text)
    return min(int(digits), 100) if digits else 0


# ---------------------------------------------------------------------------
# Shared helpers (mirrors injuries_from_images.py)
# ---------------------------------------------------------------------------

def map_to_fpl_ids(injuries: list[dict]) -> list[dict]:
    """Add fpl_id to each injury record by matching against elements.json."""
    import difflib

    if not ELEMENTS_PATH.exists():
        print(
            f"  Warning: {ELEMENTS_PATH} not found — skipping FPL ID mapping.",
            file=sys.stderr,
        )
        for r in injuries:
            r["fpl_id"] = None
        return injuries

    elements = json.loads(ELEMENTS_PATH.read_text())

    name_to_players: dict[str, list[dict]] = {}
    for el in elements:
        wn = el["web_name"]
        name_to_players.setdefault(wn, []).append(el)

    all_web_names = list(name_to_players.keys())

    teams_path = RAW_DIR / "teams.json"
    team_id_map: dict[int, str] = {}
    if teams_path.exists():
        teams = json.loads(teams_path.read_text())
        team_id_map = {t["id"]: t["name"] for t in teams}

    def same_team(el: dict, fpl_team: str) -> bool:
        name = team_id_map.get(el.get("team"), "")
        return fpl_team.lower() in name.lower() or name.lower() in fpl_team.lower()

    for record in injuries:
        player_name = record["player_name"]
        team = record.get("team", "")
        matched_id = None

        if player_name in name_to_players:
            candidates = name_to_players[player_name]
            preferred = [c for c in candidates if same_team(c, team)] if team else []
            matched_id = (preferred or candidates)[0]["id"]
        else:
            close = difflib.get_close_matches(player_name, all_web_names, n=5, cutoff=0.6)
            for cn in close:
                candidates = name_to_players[cn]
                preferred = [c for c in candidates if same_team(c, team)] if team else candidates
                if preferred:
                    matched_id = preferred[0]["id"]
                    print(f"  Fuzzy match: '{player_name}' -> '{cn}' (id={matched_id})")
                    break

        if matched_id is None:
            print(f"  Warning: no FPL match for '{player_name}' ({team})", file=sys.stderr)

        record["fpl_id"] = matched_id

    return injuries


def save_injuries(injuries: list[dict]) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "gmail_email",
        "injuries": injuries,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    return OUTPUT_PATH


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Fetching emails from premierinjuries.com...")
    bodies = fetch_emails(n=10)

    if not bodies:
        print("No emails to process. Subscribe to premierinjuries.com alerts and try again.")
        return

    print(f"  Found {len(bodies)} email(s)")

    all_records: list[dict] = []
    for i, html in enumerate(bodies, 1):
        records = parse_email_body(html)
        print(f"  Email {i}: {len(records)} player(s) parsed")
        all_records.extend(records)

    # Deduplicate: keep most recent occurrence per player name
    seen: dict[str, dict] = {}
    for record in all_records:
        seen[record["player_name"]] = record
    deduped = list(seen.values())
    print(f"  After deduplication: {len(deduped)} unique player(s)")

    deduped = map_to_fpl_ids(deduped)
    unmatched = sum(1 for r in deduped if r["fpl_id"] is None)

    path = save_injuries(deduped)
    print(f"\nSaved -> {path}")
    print(f"  Players total : {len(deduped)}")
    print(f"  FPL ID misses : {unmatched}")


if __name__ == "__main__":
    main()
