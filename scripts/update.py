#!/usr/bin/env python3
"""
Balanced advanced/basic schedule + fuzzy event deduplication.

Dedup: prevents (and cleans up) near-duplicate history entries caused by
different LLM runs describing the same event with slightly different wording.
"""
import json, os, re, time, hashlib, difflib, requests
from datetime import date, datetime, timezone, timedelta

TAVILY    = os.environ["TAVILY_API_KEY"]
GROQ      = os.environ["GROQ_API_KEY"]
CF_ACCT   = os.environ["CF_ACCOUNT_ID"]
CF_TOKEN  = os.environ["CF_API_TOKEN"]

USAGE_FILE    = "data/tavily_usage.json"
CYCLE_BUDGET  = 1000
SAFETY_BUFFER = 50

ALLOWED = [
    "reuters.com","apnews.com","afp.com","bbc.com","bbc.co.uk","aljazeera.com",
    "theguardian.com","nytimes.com","washingtonpost.com","france24.com",
    "crisisgroup.org","un.org","unhcr.org","cfr.org","who.int","icrc.org",
]

# ---------- search ----------
def tavily(query, depth="basic"):
    r = requests.post("https://api.tavily.com/search", json={
        "api_key": TAVILY, "query": query, "search_depth": depth,
        "max_results": 6, "days": 10, "include_domains": ALLOWED,
    }, timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])

# ---------- fuzzy dedup helpers ----------
def _tokens(s):
    if not s or not isinstance(s, str):
        return set()
    s = re.sub(r'[^\w\s]', ' ', s.lower())
    return set(s.split())

def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _is_duplicate_pair(a, b):
    """True if two entries describe the same event (same conflict required)."""
    if a.get('conflict') != b.get('conflict'):
        return False
    try:
        da = date.fromisoformat(a.get('date', ''))
        db = date.fromisoformat(b.get('date', ''))
    except (ValueError, TypeError):
        return False
    if abs((da - db).days) > 2:
        return False

    ta = _tokens(a.get('title', ''))
    tb = _tokens(b.get('title', ''))
    if ta and tb and _jaccard(ta, tb) >= 0.65:
        return True

    da_toks = _tokens(a.get('description', ''))
    db_toks = _tokens(b.get('description', ''))
    if da_toks and db_toks and _jaccard(da_toks, db_toks) >= 0.70:
        return True

    return False

def is_duplicate_event(new_ev, existing_history):
    for ex in existing_history:
        if _is_duplicate_pair(new_ev, ex):
            return True
    return False

def prune_history_duplicates(history):
    """Remove near-duplicates from the history array. Keeps the earliest-seen
    entry (in date order) and discards later near-duplicates."""
    if len(history) < 2:
        return history, 0
    indexed = sorted(enumerate(history), key=lambda p: (p[1].get('date', ''), p[0]))
    kept = []
    kept_indices = set()
    for orig_idx, entry in indexed:
        if not any(_is_duplicate_pair(entry, k) for k in kept):
            kept.append(entry)
            kept_indices.add(orig_idx)
    removed = len(history) - len(kept)
    # preserve original file order for stability
    return [e for i, e in enumerate(history) if i in kept_indices], removed

def recent_events_for_prompt(history, conflict_name, days=21):
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
    rows = []
    for h in history:
        if h.get('conflict') != conflict_name:
            continue
        try:
            hd = date.fromisoformat(h.get('date', ''))
        except (ValueError, TypeError):
            continue
        if hd >= cutoff:
            rows.append(f"- {h['date']}: {h.get('title', '')}")
    return "\n".join(rows) if rows else "(none recorded recently)"

# ---------- usage tracking ----------
def load_usage():
    now = datetime.now(timezone.utc)
    cycle_key = now.strftime("%Y-%m")
    try:
        with open(USAGE_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    if data.get("cycle") != cycle_key:
        data = {
            "cycle": cycle_key,
            "cycle_budget": CYCLE_BUDGET,
            "credits_used": 0,
            "advanced_days_done": 0,
            "last_advanced_date": None,
            "last_updated": now.isoformat(),
        }
    data.setdefault("advanced_days_done", 0)
    data.setdefault("last_advanced_date", None)
    return data

def save_usage(usage):
    usage["last_updated"] = datetime.now(timezone.utc).isoformat()
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f, indent=2)

def compute_advanced_this_run(usage, now, is_second_run, searches_per_run):
    if not is_second_run:
        return 0, {"reason": "first run of day → mandatory all-basic"}

    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (month_start.replace(year=now.year + 1, month=1)
                  if now.month == 12
                  else month_start.replace(month=now.month + 1))

    seconds_in_month = (next_month - month_start).total_seconds()
    progress = (now - month_start).total_seconds() / seconds_in_month

    tomorrow = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                + timedelta(days=1))
    full_days_after = max(0, (next_month - tomorrow).days)
    runs_remaining = 1 + full_days_after * 2

    budget_left = usage["cycle_budget"] - SAFETY_BUFFER - usage["credits_used"]
    min_future_cost = runs_remaining * searches_per_run
    spare = max(0, budget_left - min_future_cost)

    max_possible = full_days_after + 1
    advanced_days_total = min(spare // searches_per_run, max_possible)
    advanced_done = usage.get("advanced_days_done", 0)
    last_adv_date = usage.get("last_advanced_date")
    today_str = now.date().isoformat()

    if advanced_done >= advanced_days_total:
        return 0, {
            "reason": f"all {advanced_days_total} advanced days already used",
            "progress": progress, "spare": spare,
            "advanced_days_total": advanced_days_total,
        }

    next_threshold = (advanced_done + 1) / (advanced_days_total + 1)

    if progress >= next_threshold and last_adv_date != today_str:
        upgrades = min(searches_per_run, spare)
        usage["advanced_days_done"] = advanced_done + 1
        usage["last_advanced_date"] = today_str
        return upgrades, {
            "reason": f"ADVANCED DAY #{advanced_done + 1} of {advanced_days_total} "
                      f"(threshold {next_threshold*100:.0f}%, progress {progress*100:.0f}%)",
            "progress": progress, "spare": spare,
            "advanced_days_total": advanced_days_total,
        }

    return 0, {
        "reason": (f"basic day — next advanced at {next_threshold*100:.0f}% "
                   f"(now {progress*100:.0f}%)"),
        "progress": progress, "spare": spare,
        "advanced_days_total": advanced_days_total,
    }

# ---------- tolerant JSON parse ----------
def parse_json(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ValueError(f"expected str or dict, got {type(text)}")
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)
        s = s[1] if len(s) > 1 else s[0]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        depth, end = 0, None
        for i, ch in enumerate(s):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = i + 1
        if end:
            return json.loads(s[:end])
        raise

# ---------- time helpers ----------
def parse_time_string(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip().upper()
    if s.count(':') == 2:
        s = s.rsplit(':', 1)[0]
    m = re.match(r'^(\d{1,2}):(\d{2})\s*(AM|PM)?$', s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    ampm = m.group(3)
    if ampm:
        if ampm == 'PM' and h < 12: h += 12
        if ampm == 'AM' and h == 12: h = 0
    if h > 23 or mi > 59:
        return None
    return (h, mi)

def normalize_event_time(ev):
    if not isinstance(ev, dict):
        return ev
    raw_time = ev.get("time")
    try:
        count = int(ev.get("time_source_count") or 0)
    except (ValueError, TypeError):
        count = 0
    try:
        variance = int(ev.get("time_variance_minutes") or 999)
    except (ValueError, TypeError):
        variance = 999
    parsed = parse_time_string(raw_time)
    if not parsed:
        ev["time"] = None
        ev["time_verified"] = None
        ev.pop("time_source_count", None)
        ev.pop("time_variance_minutes", None)
        return ev
    h, mi = parsed
    time_str = f"{h:02d}:{mi:02d}"
    if count >= 2 and variance <= 25:
        ev["time"] = time_str
        ev["time_verified"] = "verified"
    elif count >= 4 and variance <= 60:
        ev["time"] = time_str
        ev["time_verified"] = "approximate"
    else:
        ev["time"] = None
        ev["time_verified"] = None
    ev.pop("time_source_count", None)
    ev.pop("time_variance_minutes", None)
    return ev

# ---------- models ----------
def groq(model, prompt, max_tokens=900, json_mode=True, retries=2):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    for attempt in range(retries + 1):
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ}", "Content-Type": "application/json"},
            json=payload, timeout=120,
        )
        if r.status_code == 429 and attempt < retries:
            print(f"  {model} 429, waiting 20s (attempt {attempt+1}/{retries})")
            time.sleep(20)
            continue
        if not r.ok:
            raise RuntimeError(f"Groq {model} HTTP {r.status_code}: {r.text[:200]}")
        return parse_json(r.json()["choices"][0]["message"]["content"])

def cloudflare(model, prompt, max_tokens=1200):
    r = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCT}/ai/run/{model}",
        headers={"Authorization": f"Bearer {CF_TOKEN}", "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Cloudflare HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"CF error: {body.get('errors')}")
    result = body["result"]["response"]
    if isinstance(result, dict):
        return result
    return parse_json(result)

# ---------- prompts ----------
SCHEMA = """{
  "current": "2-4 sentences on the present situation",
  "sources": [{"label": "Outlet: headline", "url": "https://..."}],
  "events": [
    {
      "date": "YYYY-MM-DD",
      "time": "HH:MM or null",
      "timezone": "UTC or zone abbreviation or local",
      "time_source_count": 0,
      "time_variance_minutes": 0,
      "title": "...",
      "description": "..."
    }
  ],
  "confidence": "high" | "medium" | "low"
}"""

TIME_RULES = """
TIME-VERIFICATION RULES (strict — apply to every event):
- Extract the time the EVENT itself happened, NEVER the article's publication time.
- Cluster hour:minute values; 2+ sources within 25 minutes = valid cluster.
- "time" = MEDIAN HH:MM of the largest cluster. If none, null.
- "time_source_count" = size of that cluster.
- "time_variance_minutes" = (max-min). Single source = 0. No cluster = 999.
- "timezone" = reported zone ("UTC", "EET", "local", etc.).
- DO NOT fabricate times. Missing is better than wrong.
"""

def research_prompt(conflict, results, recent_events):
    ctx = "\n".join(
        f"- {x['title']} — {x.get('content','')[:350]} ({x['url']})"
        for x in results
    )
    return f"""Today is {date.today().isoformat()}.
Conflict: {conflict['name']} (tier: {conflict['tier']}).

Verified search results from the last 10 days:
{ctx}

Events ALREADY recorded for this conflict (do NOT re-report these;
if new information refines an existing event, do NOT create a new event for it):
{recent_events}

Return ONLY valid JSON, exactly this shape:
{SCHEMA}

{TIME_RULES}

Other rules:
- Every claim in "current" must be supported by a result above.
- "sources": AT LEAST 3, up to 5. Only URLs from the list above.
- "events": 0-3 items, only developments from the last 7 days
  appearing in at least two results AND not already recorded above.
- Each event's title must be a short, unique label — never a paraphrase
  of one already recorded.
- Keep every string under 400 characters."""

def review_prompt(conflict, results, a, b, flags, recent_events):
    ctx = "\n".join(
        f"- {x['title']} — {x.get('content','')[:350]} ({x['url']})"
        for x in results
    )
    return f"""Today is {date.today().isoformat()}.
You are the REVIEWER for "{conflict['name']}". Two models disagreed.

Search results:
{ctx}

Model A:
{json.dumps(a, ensure_ascii=False)}

Model B:
{json.dumps(b, ensure_ascii=False)}

Flags: {', '.join(flags)}

Already-recorded events (do NOT re-report):
{recent_events}

Decide which claims the search results support. Output ONLY the JSON
object in this shape:
{SCHEMA}

{TIME_RULES}

If results don't resolve the dispute, set confidence to "low"."""

# ---------- structural comparison ----------
def compare(a, b):
    flags = []
    ca, cb = a.get("confidence"), b.get("confidence")
    if ca == "high" and cb == "low" or ca == "low" and cb == "high":
        flags.append("confidence mismatch (large)")
    ea = [e for e in a.get("events", []) if isinstance(e, dict) and e.get("date")]
    eb = [e for e in b.get("events", []) if isinstance(e, dict) and e.get("date")]
    if ea and eb:
        da = {e["date"] for e in ea}
        db = {e["date"] for e in eb}
        if da != db and not (da & db):
            flags.append("event-date mismatch")
    return flags

# ---------- validation ----------
def url_alive(url):
    try:
        h = requests.get(url, allow_redirects=True, timeout=6, stream=True,
                         headers={
                           "User-Agent": "Mozilla/5.0 (compatible; ConflictMap/1.0)",
                           "Accept": "text/html,application/xhtml+xml,*/*",
                           "Range": "bytes=0-1024",
                         })
        h.close()
        return h.status_code < 400 or h.status_code == 403
    except Exception:
        return False

def url_valid(url, tavily_urls):
    base = url.split('?')[0].rstrip('/')
    for tu in tavily_urls:
        if tu.split('?')[0].rstrip('/') == base:
            return True
    return url_alive(url)

def save_if_changed(path, data):
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    try:
        with open(path) as f:
            if f.read() == text:
                return False
    except FileNotFoundError:
        pass
    with open(path, "w") as f:
        f.write(text)
    return True

# ---------- main ----------
def main():
    conflicts = json.load(open("data/conflicts.json"))
    history   = json.load(open("data/history.json"))

    # Prune existing duplicates from history BEFORE doing any work
    history, pruned = prune_history_duplicates(history)
    if pruned:
        print(f"Pruned {pruned} duplicate histor{'y entry' if pruned==1 else 'y entries'} from prior runs")
    else:
        print("No duplicate history entries found")

    seen = {h["id"] for h in history}
    searches_per_run = len(conflicts)

    usage = load_usage()
    now = datetime.now(timezone.utc)
    is_second_run = (now.hour >= 12)

    advanced_count, info = compute_advanced_this_run(
        usage, now, is_second_run, searches_per_run
    )

    print(f"Tavily cycle {usage['cycle']}: {usage['credits_used']}/{usage['cycle_budget']} used")
    print(f"Run type: {'SECOND (12:00-23:59 UTC)' if is_second_run else 'FIRST (00:00-11:59 UTC)'}")
    print(f"Plan: {advanced_count} advanced + {searches_per_run - advanced_count} basic "
          f"= {advanced_count*2 + (searches_per_run - advanced_count)} credits")
    print(f"Reason: {info['reason']}")
    print("")

    save_usage(usage)

    ok = 0
    skipped_dupes = 0
    for i, c in enumerate(conflicts):
        use_adv = (i < advanced_count)
        depth = "advanced" if use_adv else "basic"
        try:
            results = tavily(f"{c['name']} conflict latest news", depth=depth)
            usage["credits_used"] += (2 if use_adv else 1)
            save_usage(usage)

            if not results:
                print(f"SKIP {c['name']}: no search results")
                continue

            tavily_urls = [x['url'] for x in results if x.get('url')]
            recent = recent_events_for_prompt(history, c['name'])
            prompt = research_prompt(c, results, recent)

            a = None
            try:
                a = cloudflare("@cf/meta/llama-3.3-70b-instruct-fp8-fast", prompt)
                time.sleep(1.0)
            except Exception as e:
                print(f"  cloudflare failed for {c['name']}: {e}")

            b = None
            try:
                b = groq("qwen/qwen3.8-27b", prompt, max_tokens=900)
                time.sleep(5.0)
            except Exception as e:
                print(f"  groq failed for {c['name']}: {e}")
                time.sleep(5.0)

            if a is None and b is None:
                print(f"SKIP {c['name']}: both researchers failed")
                continue

            if a is None:
                final, verified = b, "groq-only"
            elif b is None:
                final, verified = a, "cloudflare-only"
            else:
                flags = compare(a, b)
                if flags:
                    print(f"  review {c['name']}: {flags}")
                    try:
                        final = groq("openai/gpt-oss-120b",
                                     review_prompt(c, results, a, b, flags, recent),
                                     max_tokens=2000, json_mode=False)
                        time.sleep(5.0)
                        verified = "cloudflare+groq+groq-review"
                    except Exception as e:
                        print(f"  reviewer failed for {c['name']}: {e}; using A")
                        final, verified = a, "cloudflare+groq (no review)"
                else:
                    final, verified = a, "cloudflare+groq"

            if not isinstance(final, dict) or "current" not in final:
                print(f"SKIP {c['name']}: final output missing 'current'")
                continue

            live = [s for s in final.get("sources", [])
                    if isinstance(s, dict) and s.get("url")
                    and url_valid(s["url"], tavily_urls)]
            if len(live) < 2:
                print(f"SKIP {c['name']}: fewer than 2 live sources")
                continue

            c["current"]    = final["current"]
            c["sources"]    = live[:4]
            c["confidence"] = final.get("confidence") or "medium"
            c["verified_by"] = verified
            c["updated"]    = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ok += 1

            for ev in final.get("events", []):
                if not isinstance(ev, dict): continue
                if "date" not in ev or "title" not in ev: continue
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(ev.get("date", ""))):
                    continue
                ev = normalize_event_time(ev)
                eid = hashlib.sha1(
                    f"{c['id']}|{ev['date']}|{ev['title']}".encode()
                ).hexdigest()[:12]
                if eid in seen:
                    skipped_dupes += 1
                    continue
                candidate = {
                    "id": eid, "date": ev["date"],
                    "conflict": c["name"], "tier": c["tier"],
                    "title": ev["title"], "description": ev.get("description",""),
                    "source": "auto",
                }
                if ev.get("time") and ev.get("time_verified"):
                    candidate["time"] = ev["time"]
                    candidate["time_verified"] = ev["time_verified"]
                    if ev.get("timezone"):
                        candidate["timezone"] = ev["timezone"]

                # Fuzzy dedup against existing history
                if is_duplicate_event(candidate, history):
                    skipped_dupes += 1
                    print(f"  dup skipped for {c['name']}: {ev['title'][:60]}")
                    continue

                seen.add(eid)
                history.append(candidate)

            print(f"OK {c['name']} ({depth}, {len(live)} src, {c['confidence']})")

        except Exception as e:
            print(f"FAIL {c['name']}: {e}")

    # Prune again in case new events happened to be near-duplicates of each other
    history, pruned2 = prune_history_duplicates(history)
    if pruned2:
        print(f"Post-run cleanup: pruned {pruned2} additional duplicate(s)")

    history.sort(key=lambda h: h["date"], reverse=True)
    c_changed = save_if_changed("data/conflicts.json", conflicts)
    h_changed = save_if_changed("data/history.json",   history)
    print(f"\n{ok}/{len(conflicts)} refreshed; "
          f"conflicts {'updated' if c_changed else 'unchanged'}, "
          f"history {'updated' if h_changed else 'unchanged'}")
    print(f"Fuzzy-dedup skips: {skipped_dupes} new event(s) rejected as duplicates")
    print(f"Tavily total: {usage['credits_used']}/{usage['cycle_budget']} credits, "
          f"advanced days done: {usage['advanced_days_done']}")

if __name__ == "__main__":
    main()
