#!/usr/bin/env python3
"""
Stage 2: resolve dates for queued events using ONLY free APIs.

Zero credit card. Zero paid tiers. No Tavily credits.

Sources (in order of preference):
  1. DuckDuckGo via `ddgs` — no key, no signup
  2. GDELT DOC 2.0        — no key, no signup
  3. GNews                — email-only free tier (100 req/day)

Consensus rule: a date is promoted to history.json only when at least
two of the three sources agree within a 1-day tolerance AND that date
is within ±14 days of the LLM's original guess.
"""
import json, os, re, time, requests
from datetime import date, datetime, timezone, timedelta

GNEWS_KEY = os.environ.get("GNEWS_API_KEY", "")  # optional fallback

PENDING_FILE = "data/pending_dates.json"
HISTORY_FILE = "data/history.json"

# ---------- 1. DuckDuckGo (primary) ----------
def ddg_search(query):
    """News search via ddgs. Returns list of {title, url, date, source}."""
    try:
        try:
            from ddgs import DDGS          # new package name (2026)
        except ImportError:
            from duckduckgo_search import DDGS  # legacy name
        out = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=15, timelimit="m"):
                d = r.get("date") or ""
                try:
                    dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
                    out.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "date": dt.date(),
                        "source": "ddg",
                    })
                except (ValueError, TypeError):
                    continue
        return out
    except Exception as e:
        print(f"  DDG error: {e}")
        return []

# ---------- 2. GDELT DOC 2.0 (secondary, no key ever) ----------
def gdelt_search(query, after_date, before_date):
    """Query GDELT DOC 2.0. Returns list of {title, url, date, source}."""
    try:
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": 30,
            "format": "json",
            "startdatetime": after_date.strftime("%Y%m%d000000"),
            "enddatetime": before_date.strftime("%Y%m%d235959"),
        }
        r = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params, timeout=30,
        )
        if not r.ok:
            print(f"  GDELT HTTP {r.status_code}")
            return []
        data = r.json()
        out = []
        for art in (data.get("articles") or []):
            sd = art.get("seendate") or ""
            try:
                dt = datetime.strptime(sd[:8], "%Y%m%d")
                out.append({
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "date": dt.date(),
                    "source": "gdelt",
                })
            except (ValueError, TypeError):
                continue
        return out
    except Exception as e:
        print(f"  GDELT error: {e}")
        return []

# ---------- 3. GNews (tertiary, email-only free tier) ----------
def gnews_search(query, after_date, before_date):
    """Optional fallback. Skipped silently if no key."""
    if not GNEWS_KEY:
        return []
    try:
        r = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": query,
                "from": after_date.isoformat() + "T00:00:00Z",
                "to": before_date.isoformat() + "T23:59:59Z",
                "lang": "en",
                "max": 10,
                "apikey": GNEWS_KEY,
            }, timeout=30,
        )
        if not r.ok:
            print(f"  GNews HTTP {r.status_code}")
            return []
        out = []
        for art in (r.json().get("articles") or []):
            try:
                dt = datetime.fromisoformat(
                    art["publishedAt"].replace("Z", "+00:00"))
                out.append({
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "date": dt.date(),
                    "source": "gnews",
                })
            except (ValueError, KeyError, TypeError):
                continue
        return out
    except Exception as e:
        print(f"  GNews error: {e}")
        return []

# ---------- consensus ----------
def consensus(hits, llm_date, tolerance_days=14):
    """Requires ≥2 independent sources agreeing within 1 day.
    Returns (chosen_date, confidence, reason)."""
    try:
        llm_d = date.fromisoformat(llm_date)
    except (ValueError, TypeError):
        return None, "none", "invalid LLM date"

    # Filter out candidates too far from LLM guess (likely unrelated)
    candidates = {}
    for h in hits:
        if abs((h["date"] - llm_d).days) > tolerance_days:
            continue
        key = h["date"]
        if key not in candidates:
            candidates[key] = {"sources": set(), "count": 0}
        candidates[key]["sources"].add(h["source"])
        candidates[key]["count"] += 1

    if not candidates:
        return None, "none", "no candidates within ±14d of LLM guess"

    # Score each candidate by number of DISTINCT sources + hit count
    scored = []
    for d, info in candidates.items():
        source_count = len(info["sources"])
        scored.append((d, source_count, info["count"]))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)

    best_date, src_count, hit_count = scored[0]

    # Consensus requires ≥2 DISTINCT sources
    if src_count >= 2:
        # Bonus: if it also matches the LLM guess exactly
        if best_date == llm_d:
            return best_date, "high", \
                f"consensus from {src_count} sources ({hit_count} hits), matches LLM"
        return best_date, "medium", \
            f"consensus from {src_count} sources ({hit_count} hits), LLM said {llm_date}"

    # Single-source with many hits is weak — reject
    return None, "low", \
        f"only {src_count} source ({hit_count} hits) — not enough for consensus"

# ---------- main ----------
def main():
    if not os.path.exists(PENDING_FILE):
        print("No pending_dates.json — nothing to verify.")
        return
    try:
        pending = json.load(open(PENDING_FILE))
    except json.JSONDecodeError:
        print("pending_dates.json invalid — skipping.")
        return
    if not pending:
        print("Pending queue is empty.")
        return

    history = json.load(open(HISTORY_FILE))
    resolved = 0
    unresolved = 0
    remaining = []

    for item in pending:
        if item.get("status") != "pending":
            remaining.append(item)
            continue

        title = item.get("title", "")
        conflict = item.get("conflict", "")
        llm_date = item.get("llm_date", "")

        try:
            llm_d = date.fromisoformat(llm_date)
        except (ValueError, TypeError):
            item["status"] = "invalid_llm_date"
            unresolved += 1
            remaining.append(item)
            continue

        after = llm_d - timedelta(days=7)
        before = llm_d + timedelta(days=7)
        query = f"{conflict} {title}"[:200]

        print(f"\nVerifying: {conflict} — {title[:60]} (LLM: {llm_date})")

        hits = []
        # 1. DDG (primary — no signup)
        ddg = ddg_search(query)
        print(f"  DDG: {len(ddg)} hits")
        hits.extend(ddg)

        # 2. GDELT (secondary — no signup) — throttle to 1 req / 6 s
        time.sleep(1.5)
        gd = gdelt_search(query, after, before)
        print(f"  GDELT: {len(gd)} hits")
        hits.extend(gd)

        # 3. GNews (tertiary — email-only, optional)
        if GNEWS_KEY:
            gn = gnews_search(query, after, before)
            print(f"  GNews: {len(gn)} hits")
            hits.extend(gn)

        chosen, conf, reason = consensus(hits, llm_date)
        print(f"  → {chosen} ({conf}) — {reason}")

        if chosen and conf in ("high", "medium"):
            slug = re.sub(r'[^a-z0-9]+', '-', conflict.lower()).strip('-')
            new_id = f"verified-{slug}-{chosen.isoformat()}-{len(history)}"
            entry = {
                "id": new_id,
                "date": chosen.isoformat(),
                "conflict": conflict,
                "tier": item.get("tier", "war"),
                "title": title,
                "description": item.get("description", ""),
                "source": "auto-verified-free",
                "verified_date_method": f"free-consensus ({conf})",
                "original_llm_date": llm_date,
            }
            if item.get("llm_time"):
                entry["time"] = item["llm_time"]
                entry["time_verified"] = "unverified_from_stage1"
            if item.get("llm_timezone"):
                entry["timezone"] = item["llm_timezone"]
            history.append(entry)
            resolved += 1
            item["status"] = "resolved"
        else:
            item["status"] = "unresolved"
            item["verification_reason"] = reason
            unresolved += 1

        remaining.append(item)
        time.sleep(1.5)  # be polite between events

    history.sort(key=lambda h: h.get("date", ""), reverse=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(PENDING_FILE, "w") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)

    print(f"\nResolved {resolved}; unresolved {unresolved}.")
    print(f"pending_dates.json now has {len(remaining)} entry/entries.")

if __name__ == "__main__":
    main()
