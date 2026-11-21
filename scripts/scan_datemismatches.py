#!/usr/bin/env python3
"""
Diagnostic scan for date anomalies in the repository's JSON data files.

Usage:
    python scripts/scan_datemismatches.py
"""

import json
import sys
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = REPO_ROOT / "data" / "history.json"
CONFLICTS_PATH = REPO_ROOT / "data" / "conflicts.json"

# Allow events timestamped up to one day in the future, e.g. because of
# timezone differences or publication-time ambiguity.
FUTURE_DATE_TOLERANCE_DAYS = 1


def load_json(path: Path):
    """Load a JSON file, returning None for a missing or invalid file."""
    if not path.exists():
        print(f"Warning: {path} not found.")
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: {path} is not valid JSON: {exc}")
        return None
    except OSError as exc:
        print(f"Error: could not read {path}: {exc}")
        return None


def as_records(data):
    """Return a list of records from either a list or a keyed dictionary."""
    if data is None:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]

    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]

    print(f"Warning: unexpected JSON structure: {type(data).__name__}")
    return []


def parse_event_date(value):
    """Parse a YYYY-MM-DD event date, returning None when invalid."""
    if not isinstance(value, str):
        return None

    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def parse_source_timestamp(value):
    """Parse an ISO timestamp and normalize it to a UTC date."""
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc).date()


def scan_history(run_date: date) -> int:
    history = load_json(HISTORY_PATH)
    records = as_records(history)
    mismatches = 0

    print(f"Scanning {len(records)} historical events (today: {run_date})...\n")
    print(f"{'DATE':<12} | {'CONFLICT':<25} | ISSUE / TITLE")
    print("-" * 100)

    for record in records:
        event_date_text = record.get("date", "")
        conflict = str(record.get("conflict", "Unknown"))
        title = str(record.get("title", "No title"))
        event_date = parse_event_date(event_date_text)

        if event_date is None:
            issue = "[INVALID DATE FORMAT]"
            mismatches += 1
        elif event_date > run_date + timedelta(days=FUTURE_DATE_TOLERANCE_DAYS):
            issue = "[FUTURE DATE]"
            mismatches += 1
        else:
            continue

        print(
            f"{event_date_text:<12} | {conflict[:25]:<25} | "
            f"{issue} {title[:50]}"
        )

    print("-" * 100)
    print(f"History scan complete. Flagged {mismatches} entries.\n")
    return mismatches


def scan_conflicts(run_date: date) -> int:
    conflicts = load_json(CONFLICTS_PATH)
    records = as_records(conflicts)
    mismatches = 0

    print("Scanning conflicts.json latest source timestamps...")

    for conflict in records:
        name = str(conflict.get("name", "Unknown"))
        latest_source_time = conflict.get("latest_source_time")

        if not latest_source_time:
            continue

        source_date = parse_source_timestamp(latest_source_time)

        if source_date is None:
            print(f"Conflict '{name}' has an invalid source timestamp: {latest_source_time}")
            mismatches += 1
        elif source_date > run_date + timedelta(days=FUTURE_DATE_TOLERANCE_DAYS):
            print(
                f"Conflict '{name}' has a future source timestamp: "
                f"{latest_source_time}"
            )
            mismatches += 1

    print(f"Conflict scan complete. Flagged {mismatches} entries.\n")
    return mismatches


def main() -> int:
    run_date = datetime.now(timezone.utc).date()

    history_issues = scan_history(run_date)
    conflict_issues = scan_conflicts(run_date)
    total_issues = history_issues + conflict_issues

    print("=" * 100)
    print(
        f"Scan complete. Flagged {total_issues} potential "
        f"anomaly/mismatch entr{'y' if total_issues == 1 else 'ies'}."
    )

    # Use exit code 1 in CI so a future-date or malformed-data problem is visible.
    return 1 if total_issues else 0


if __name__ == "__main__":
    sys.exit(main())
