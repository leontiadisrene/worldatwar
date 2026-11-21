#!/usr/bin/env python3
"""
Stage 2: resolve dates for events queued by update.py.

Uses Perplexity (free credits) and NewsAPI.ai (2,000 free searches) to
determine the most likely real date for each pending event. Cross-checks
the two sources; only promotes an event to history.json if both agree
or one provides overwhelming evidence.

Does NOT touch Tavily.
"""
import json, os, re, time, requests
from datetime import date, datetime, timezone, timedelta

PERPLEXITY_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
NEWSAPI_KEY    = os.environ.get("NEWSAPI_AI_KEY", "")

PENDING_FILE   = "data/pending_dates.json"
HISTORY_FILE   = "data/history.json"

# ---------- Perplexity ----------
def perplexity_search(query, after_date, before_date):
    """Call Perplexity's search API with a publication-date window.
    Returns list of {title, url, date, snippet} or []."""
    if not PERPLEXITY_KEY:
        return []
    try:
        r = requests.post(
            "https://api.perplexity.ai/search",
            headers={"Authorization": f"Bearer {PERPLEXITY_KEY}",
                     "Content-Type": "application/json"},
            json={
                "query": query,
                "search_after_date_filter": after_date.strftime("%m/%d/%Y"),
                "search_before_date_filter": before_date.strftime("%m/%d/%Y"),
                "max_results": 10,
            },
            timeout=45,
        )
        if not r.ok:
            print(f"  Perplexity HTTP {r.status_code}: {r.text[:160]}")
            return []
        data = r.json()
        items = data.get("results") or data.get("search_results") or []
        out = []
        for it in items:
            d = it.get("date") or it.get("published_date") or ""
            try:
                dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
                out.append({
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "date": dt.date(),
                    "snippet": it.get("snippet", ""),
                })
            except (ValueError, TypeError):
                continue
        return out
    except Exception as e:
        print(f"  Perplexity error: {e}")
        return []

# ---------- NewsAPI.ai ----------
def newsapi_search(query, after_date, before_date):
    """Call NewsAPI.ai with a date window. Returns list of article dicts."""
    if not NEWSAPI_KEY:
        return []
    try:
        r = requests.get(
            "https://eventregistry.org/api/v1/article/getArticles",
            params={
                "query": query,
                "dateStart": after_date.isoformat(),
                "dateEnd": before_date.isoformat(),
                "lang": "eng",
                "articlesCount": 20,
                "articlesSortBy": "date",
                "resultType": "articles",
                "apiKey": NEWSAPI_KEY,
            },
            timeout=45,
        )
        if not r.ok:
            print(f"  NewsAPI.ai HTTP {r.status_code}: {r.text[:160]}")
            return []
        data = r.json()
        arts = (data.get("articles") or {}).get("results") or []
        out = []
        for a in arts:
            dt_str = a.get("dateTimePub") or a.get("date") or ""
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                out.append({
                    "title": a.get("title", ""),
                    "url": a.get("url", ""),
                    "date": dt.date(),
                    "source": (a.get("source") or {}).get("title", ""),
                })
            except (ValueError, TypeError):
                continue
        return out
    except Exception as e:
        print(f"  NewsAPI.ai error: {e}")
        return []

# ---------- date consensus ----------
def extract_dates_from_text(text):
    """Find YYYY-MM-DD, 'on 14 November', 'Nov 14', etc. Returns set of dates."""
    dates = set()
    # ISO dates
    for m in re.finditer(r'(20\d{2})-(\d{2})-(\d{2})', text):
        try:
            dates.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
        except ValueError:
            pass
    # Day month year
    for m in re.finditer(r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})', text, re.IGNORECASE):
        try:
            d = datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y").date()
            dates.add(d)
        except ValueError:
            pass
    return dates

def consensus_date(perplexity_hits, newsapi_hits, llm_date_str):
    """Determine the most likely real date.
    Returns (chosen_date, confidence, reason)."""
    llm_date = None
    try:
        llm_date = date.fromisoformat(llm_date_str)
    except (ValueError, TypeError):
        pass

    # Collect candidate dates from both sources
    cand = {}
    def add(d, weight):
        if d and llm_date and abs((d - llm_date).days) > 14:
            return  # too far from LLM guess — likely unrelated
        cand[d] = cand.get(d, 0) + weight

    for h in perplexity_hits:
        add(h["date"], 2)
        for d in extract_dates_from_text(h["snippet"]):
            add(d, 1)
    for h in newsapi_hits:
        add(h["date"], 2)

    if not cand:
        return None, "none", "no candidate dates from either source"

    # Weighted consensus: prefer dates that appear in both sources
    best_date = max(cand, key=cand.get)
    best_score = cand[best_date]

    # Confidence tiers
    if best_score >= 4 and llm_date and best_date == llm_date:
        return best_date, "high", f"consensus ({best_score} pts) matches LLM guess"
    if best_score >= 4:
        return best_date, "medium", f"consensus ({best_score} pts) differs from LLM ({llm_date})"
    if best_score >= 2:
        return best_date, "low", f"weak consensus ({best_score} pts)"
    return None, "none", "insufficient evidence"

# ---------- main ----------
def main():
    if not os.path.exists(PENDING_FILE):
        print("No pending_dates.json — nothing to verify.")
        return
    try:
        pending = json.load(open(PENDING_FILE))
    except json.JSONDecodeError:
        print("pending_dates.json is invalid JSON — skipping.")
        return
    if not pending:
        print("Pending queue is empty.")
        return

    history = json.load(open(HISTORY_FILE))
    seen_ids = {h.get("id") for h in history}
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

        # Search window: 7 days before + 7 days after the LLM guess
        after = llm_d - timedelta(days=7)
        before = llm_d + timedelta(days=7)
        query = f"{conflict} {title}"

        print(f"\nVerifying: {conflict} — {title[:60]} (LLM: {llm_date})")

        p_hits = perplexity_search(query, after, before)
        n_hits = newsapi_search(query, after, before)

        chosen, conf, reason = consensus_date(p_hits, n_hits, llm_date)
        print(f"  Perplexity: {len(p_hits)} hits | NewsAPI: {len(n_hits)} hits")
        print(f"  → {chosen} ({conf}) — {reason}")

        if chosen and conf in ("high", "medium"):
            # Promote to history.json
            new_id = f"verified-{conflict.lower().replace(' ','-')}-{chosen.isoformat()}-{len(history)}"
            entry = {
                "id": new_id,
                "date": chosen.isoformat(),
                "conflict": conflict,
                "tier": item.get("tier", "war"),
                "title": title,
                "description": item.get("description", ""),
                "source": "auto-verified",
                "verified_date_method": f"perplexity+newsapi ({conf})",
                "original_llm_date": llm_date,
            }
            if item.get("llm_time"):
                entry["time"] = item["llm_time"]
                entry["time_verified"] = "unverified_from_stage1"
            if item.get("llm_timezone"):
                entry["timezone"] = item["llm_timezone"]
            if new_id not in seen_ids:
                history.append(entry)
                seen_ids.add(new_id)
                resolved += 1
            item["status"] = "resolved"
        else:
            item["status"] = "unresolved"
            item["verification_reason"] = reason
            unresolved += 1

        remaining.append(item)
        time.sleep(1.5)  # be kind to free tiers

    # Write back
    history.sort(key=lambda h: h.get("date", ""), reverse=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(PENDING_FILE, "w") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)

    print(f"\nResolved {resolved} event(s); {unresolved} left unresolved.")
    print(f"pending_dates.json now has {len(remaining)} entry/entries.")

if __name__ == "__main__":
    main()
