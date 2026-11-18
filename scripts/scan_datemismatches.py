#!/usr/bin/env python3
"""
Diagnostic scan for existing date mismatches in local JSON data files
(located in scripts/ directory).
"""
import json
from datetime import date, datetime, timezone, timedelta

def scan_existing_data():
    try:
        with open("../data/history.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        print("Error: ../data/history.json not found.")
        return

    run_date = datetime.now(timezone.utc).date()
    mismatches_found = 0

    print(f"Scanning {len(history)} historical events for potential anomalies (Today: {run_date})...\n")
    print(f"{'DATE':<12} | {'CONFLICT':<25} | {'TITLE'}")
    print("-" * 80)

    for h in history:
        ev_date_str = h.get("date", "")
        conflict = h.get("conflict", "Unknown")
        title = h.get("title", "No title")

        try:
            ev_d = date.fromisoformat(ev_date_str)
        except (ValueError, TypeError):
            print(f"{ev_date_str:<12} | {conflict[:25]:<25} | [INVALID DATE FORMAT] {title[:40]}")
            mismatches_found += 1
            continue

        # Check for future dates or suspicious Nov 18 entries (like the drone attack mismatch)
        is_future = ev_d > run_date + timedelta(days=1)
        is_suspicious_nov_18 = (ev_date_str == "2026-11-18")

        if is_future:
            print(f"{ev_date_str:<12} | {conflict[:25]:<25} | [FUTURE DATE] {title[:40]}")
            mismatches_found += 1
        elif is_suspicious_nov_18:
            print(f"{ev_date_str:<12} | {conflict[:25]:<25} | [POSSIBLE DATE MISMATCH] {title[:40]}")
            mismatches_found += 1

    # Also check active conflicts summaries
    try:
        with open("../data/conflicts.json", "r") as f:
            conflicts = json.load(f)
        
        print("\nScanning conflicts.json latest source timestamps...")
        for c in conflicts:
            name = c.get("name", "Unknown")
            latest_src = c.get("latest_source_time")
            if latest_src:
                try:
                    src_d = datetime.fromisoformat(latest_src).date()
                    if src_d > run_date + timedelta(days=1):
                        print(f"Conflict '{name}' has future source time: {latest_src}")
                        mismatches_found += 1
                except ValueError:
                    pass
    except FileNotFoundError:
        pass

    print("-" * 80)
    print(f"Scan complete. Flagged {mismatches_found} potential anomaly/mismatch entry/entries.")

if __name__ == "__main__":
    scan_existing_data()
