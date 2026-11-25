#!/usr/bin/env python3
"""
Balanced advanced/basic schedule + fuzzy/LLM duplicate handling
+ tier-aware source weighting + perspective and escalation fields.

Duplicate handling layers:
  1. Prompt injection — models see what's already recorded.
  2. LLM merge — same-date near-duplicates and same-incident multi-angle
     accounts are combined into one entry.
  3. Fuzzy prune — high-similarity cross-day repeats are dropped.

Tier-aware trust layers (Options 1+2):
  1. Prompt injection — SOURCE_TIERS + SOURCE_WEIGHTING_RULES tell the
     model how to weight sources when they disagree.
  2. Post-processing — adjust_confidence() enforces a confidence floor:
     no `high` without a Tier 1 source; no `medium` if only Tier 3 cited.

Perspective + Escalation:
  - perspective: descriptive (framing divergence Tier 1 vs Tier 3), no smoothing
  - escalation: EWMA + ±12 cap + confidence dampener (smooth_escalation)

Date guards (event insertion, applied in this order):
  - URL override: correct dates using article URL paths or Tavily metadata
  - Guard 1 (pre-flight): reject events more than +1 day in the future
    (allows timezone slips). Runs BEFORE Guard 0 so impossible dates never
    consume external verifier (scripts/verify_dates_free.py) queries.
  - Guard 0: queue event to data/pending_dates.json for external verification
    (scripts/verify_dates_free.py) if unquoted AND not rescued by URL override.
  - Guard 2: reject events where the source article was published BEFORE the
    claimed event date (impossible — can't report on something that hasn't
    happened). Tolerates ±1 day for timezone differences.
  - Guard 3: reject stale events older than 14 days (backstop for wrong-month errors)
"""
import json, os, re, time, hashlib, random, requests
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from urllib.parse import urlparse

import ceasefire_indicator as cfi  # scripts/ceasefire_indicator.py (same folder)

TAVILY    = os.environ["TAVILY_API_KEY"]
GROQ      = os.environ["GROQ_API_KEY"]
CF_ACCT   = os.environ["CF_ACCOUNT_ID"]
CF_TOKEN  = os.environ["CF_API_TOKEN"]
# Optional: independent fallback model used only when Groq fails. If it is not
# set, the fallback fails gracefully (that run counts as a single-model run).
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

USAGE_FILE    = "data/tavily_usage.json"
CYCLE_BUDGET  = 1000
SAFETY_BUFFER = 50

# ---------- tiered allowlist ----------
TIER_1_DOMAINS = [
    "reuters.com", "apnews.com", "afp.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "welt.de",
    "ctvnews.ca", "thestar.com",
]

TIER_2_DOMAINS = [
    "crisisgroup.org", "un.org", "unhcr.org", "cfr.org", "who.int",
    "icrc.org", "hrw.org", "amnesty.org",
    "theamericanconservative.com", "longwarjournal.org", "nationalinterest.org",
    "warontherocks.com", "henryjacksonsociety.org", "worldview.stratfor.com",
]

TIER_3_DOMAINS = [
    "kyivindependent.com", "kyivpost.com", "haaretz.com", "timesofisrael.com",
    "middleeasteye.net", "sudantribune.com", "thehindu.com", "dawn.com",
    "scmp.com", "arabnews.com", "gulfnews.com", "radiosvoboda.org",
    "africanews.com", "allafrica.com", "al-monitor.com", "jpost.com",
    "thediplomat.com", "asiatimes.com", "japantimes.co.jp", "the-japan-news.com",
    "aljazeera.com", "france24.com", "rfi.fr", "dw.com", "channelnewsasia.com",
]

ALLOWED = TIER_1_DOMAINS + TIER_2_DOMAINS + TIER_3_DOMAINS

SOURCE_TIERS = {}
for _d in TIER_1_DOMAINS: SOURCE_TIERS[_d] = 1
for _d in TIER_2_DOMAINS: SOURCE_TIERS[_d] = 2
for _d in TIER_3_DOMAINS: SOURCE_TIERS[_d] = 3

def tier_of(url):
    if not url or not isinstance(url, str):
        return None
    try:
        host = urlparse(url).hostname or ''
    except Exception:
        return None
    host = host.lower()
    if host.startswith('www.'):
        host = host[4:]
    for d, t in SOURCE_TIERS.items():
        if host == d or host.endswith('.' + d):
            return t
    return None

def domain_in_tier(url, tier):
    return tier_of(url) == tier

def adjust_confidence(confidence, sources):
    counts = {1: 0, 2: 0, 3: 0, 'unknown': 0}
    for s in sources:
        t = tier_of(s.get('url'))
        if t in (1, 2, 3):
            counts[t] += 1
        else:
            counts['unknown'] += 1

    conf = confidence or 'medium'
    tier1 = counts[1]
    tier2 = counts[2]

    if tier1 == 0 and tier2 == 0:
        if conf != 'low':
            return 'low', f'no Tier 1/2 sources (Tier 3: {counts[3]}, unknown: {counts["unknown"]})'
        return conf, None

    if tier1 == 0 and conf == 'high':
        return 'medium', f'no Tier 1 source (Tier 2: {tier2}, Tier 3: {counts[3]})'

    return conf, None

# ---------- search ----------
def tavily(query, depth="basic"):
    r = requests.post("https://api.tavily.com/search", json={
        "api_key": TAVILY,
        "query": query,
        "search_depth": depth,
        "max_results": 10,
        "time_range": "week",
        "include_domains": ALLOWED,
        "include_published_date": True
    }, timeout=60)
    r.raise_for_status()
    return r.json().get("results", [])

# ---------- fuzzy helpers ----------
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
    return [e for i, e in enumerate(history) if i in kept_indices], removed

# ---------- same-date merge ----------
def find_same_day_groups(history, days_back=21):
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=days_back)
    by_key = defaultdict(list)
    for idx, e in enumerate(history):
        if e.get('source') == 'gemini':
            continue
        try:
            ed = date.fromisoformat(e.get('date', ''))
        except (ValueError, TypeError):
            continue
        if ed < cutoff:
            continue
        key = (e.get('conflict', ''), e.get('date', ''))
        by_key[key].append(idx)
    return [indices for indices in by_key.values() if len(indices) >= 2]

# The merge prompt used to carry EVERY same-day group in a single call. With
# dozens of groups the reply came back cut off / malformed, the whole merge
# failed, and the groups piled up run after run. They are now reviewed in
# small batches so one bad reply only costs one batch.
MERGE_BATCH_SIZE = 8       # same-day groups sent to the model per merge call
MERGE_BATCH_PAUSE = 3.0    # seconds to wait between merge calls

def _merge_batch(groups, history, label=""):
    """Review ONE batch of same-day groups. Returns a list of merges (possibly
    empty), or None if the model call or its reply could not be used. Group
    ids in the prompt are local to the batch (0..n-1)."""
    groups_data = []
    for i, indices in enumerate(groups):
        entries = [history[idx] for idx in indices]
        groups_data.append({
            "group": i,
            "conflict": entries[0].get('conflict', ''),
            "date": entries[0].get('date', ''),
            "entries": [
                {"title": e.get('title', ''), "description": e.get('description', '')}
                for e in entries
            ],
        })

    prompt = f"""You are consolidating a conflict-tracker's event log.
Below are groups of events recorded on the SAME day for the SAME conflict.
Some groups contain duplicates or opposing/different-side accounts of the SAME
incident; others contain genuinely distinct events that happened on the same day.

For each group:
- If ALL entries describe the SAME event — including different angles, sides,
  claims, or counter-claims regarding the SAME underlying incident on that
  day — output merge=true with a single merged entry combining unique factual
  details and perspectives from all of them, with clear attribution ("X said",
  "Y reported") where claims differ. Do NOT invent facts.
  Max 80 chars for title, 1-3 sentences for description.
- If entries describe genuinely DISTINCT incidents on the same day (different
  underlying events/locations/actions) → output merge=false for the whole group.

Return ONLY a JSON array in this exact shape:
[
  {{"group": 0, "merge": true, "title": "...", "description": "..."}},
  {{"group": 1, "merge": false}}
]

Groups:
{json.dumps(groups_data, indent=2, ensure_ascii=False)}
"""

    try:
        result = groq_with_fallback(
            "openai/gpt-oss-120b", prompt, max_tokens=2500, json_mode=False
        )
    except Exception as e:
        print(f"  merge LLM failed{label}: {e}")
        return None

    if isinstance(result, dict):
        for key in ("results", "merges", "groups"):
            if key in result and isinstance(result[key], list):
                result = result[key]
                break
    if not isinstance(result, list):
        print(f"  merge LLM returned unexpected type{label}: {type(result).__name__}")
        return None

    merges = []
    for item in result:
        if not isinstance(item, dict):
            continue
        if not item.get("merge"):
            continue
        gi = item.get("group")
        if not isinstance(gi, int) or gi < 0 or gi >= len(groups):
            continue
        merges.append({
            "group": gi,
            "indices": groups[gi],
            "title": (item.get("title") or "").strip(),
            "description": (item.get("description") or "").strip(),
        })
    return merges

def merge_groups_with_llm(groups, history):
    """Review all same-day groups, MERGE_BATCH_SIZE at a time. A batch that
    fails is reported and skipped; the other batches still go ahead."""
    if not groups:
        return []
    n_batches = (len(groups) + MERGE_BATCH_SIZE - 1) // MERGE_BATCH_SIZE
    merges = []
    failed = 0
    for b in range(n_batches):
        chunk = groups[b * MERGE_BATCH_SIZE:(b + 1) * MERGE_BATCH_SIZE]
        if b > 0:
            time.sleep(MERGE_BATCH_PAUSE)
        got = _merge_batch(chunk, history, label=f" (batch {b + 1}/{n_batches})")
        if got is None:
            failed += 1
        else:
            merges.extend(got)
    if failed:
        print(f"  merge: {failed} of {n_batches} batch(es) FAILED - "
              f"those groups were not reviewed this run")
    return merges

def apply_merges(history, merges):
    remove_indices = set()
    for m in merges:
        indices = m["indices"]
        entries = [history[idx] for idx in indices]

        def time_score(e):
            v = e.get('time_verified')
            tier = 2 if v == 'verified' else 1 if v == 'approximate' else 0
            try:
                cnt = int(e.get('time_source_count') or 0)
            except (ValueError, TypeError):
                cnt = 0
            return (tier, cnt)

        best = max(entries, key=time_score)
        final_title = m["title"] or best.get('title', '')
        final_desc = m["description"] or best.get('description', '')

        merged = {
            "id": hashlib.sha1(
                f"{best.get('conflict','')}|{best.get('date','')}|{final_title}".encode()
            ).hexdigest()[:12],
            "date": best.get('date', ''),
            "conflict": best.get('conflict', ''),
            "tier": best.get('tier', 'war'),
            "title": final_title,
            "description": final_desc,
            "source": "merged",
        }
        if best.get('time') and best.get('time_verified'):
            merged["time"] = best["time"]
            merged["time_verified"] = best["time_verified"]
            if best.get('timezone'):
                merged["timezone"] = best["timezone"]

        print(f"  merged {len(indices)} entries → \"{final_title[:70]}\"")
        history[indices[0]] = merged
        for idx in indices[1:]:
            remove_indices.add(idx)

    return [e for i, e in enumerate(history) if i not in remove_indices]

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

def earliest_source_time_from_results(results):
    """Extract the earliest published_date from Tavily results
    to find when the story actually broke.
    Returns (datetime_utc_or_None, source_metadata_or_None)."""
    earliest = None
    earliest_meta = None
    for r in results:
        pd = r.get('published_date') or r.get('published') or ''
        if not pd or not isinstance(pd, str):
            continue
        try:
            dt = datetime.fromisoformat(pd.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            elif dt.tzinfo != timezone.utc:
                dt = dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            try:
                d = date.fromisoformat(pd[:10])
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
        if earliest is None or dt < earliest:
            earliest = dt
            earliest_meta = {
                "url": r.get('url', ''),
                "title": (r.get('title') or '')[:100],
            }
    return earliest, earliest_meta

def extract_date_from_url(url):
    """Extract YYYY-MM-DD from news URL paths (e.g. /2026-11-16/)."""
    if not url or not isinstance(url, str):
        return None
    m = re.search(r'/(\d{4})-(\d{2})-(\d{2})/', url)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None

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
        return 0, {"reason": f"all {advanced_days_total} advanced days already used",
                   "progress": progress, "spare": spare,
                   "advanced_days_total": advanced_days_total}
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
    # A variance of 0 is valid (sources agree exactly), so only fall back to
    # 999 when the value is missing -- `or 999` would turn 0 into 999.
    raw_variance = ev.get("time_variance_minutes")
    try:
        variance = int(raw_variance) if raw_variance is not None else 999
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
    elif count >= 2 and variance <= 60:
        ev["time"] = time_str
        ev["time_verified"] = "approximate"
    else:
        ev["time"] = None
        ev["time_verified"] = None
    ev.pop("time_source_count", None)
    ev.pop("time_variance_minutes", None)
    return ev

# ---------- escalation smoothing ----------
def smooth_escalation(raw_score, prev_score, confidence, source_count):
    """Stabilise the escalation score across runs.
    - EWMA: 50/50 blend with previous score (when prev exists)
    - Hard cap: ±12 per run
    - Confidence dampener: low → 0.5×, medium → 0.75×
    - Thin-source dampener: <3 sources → 0.7×
    """
    try:
        raw = int(raw_score)
    except (ValueError, TypeError):
        return None
    raw = max(0, min(100, raw))

    # First run for this conflict — seed with the raw score
    if prev_score is None:
        return raw

    delta = raw - prev_score
    delta = max(-12, min(12, delta))
    if confidence == 'low':
        delta = round(delta * 0.35)
    elif confidence == 'medium':
        delta = round(delta * 0.7)
    if source_count < 3:
        delta = round(delta * 0.7)

    return max(0, min(100, prev_score + delta))

# ---------- models ----------
# Free fallback: Mistral, a model family independent of both Cloudflare's Llama
# (researcher A) and Groq's models (researcher B). It is used only after Groq
# retries are exhausted. It must NOT be the same model as researcher A: if it
# were, one model answering twice would look like two models agreeing in the
# ceasefire indicator. Its output passes through the same downstream
# validation as the primary model.
FALLBACK_PROVIDER = "mistral"
FALLBACK_MODEL = "mistral-small-latest"

GROQ_MAX_ATTEMPTS = 5          # 1 initial request + 4 retries
GROQ_INITIAL_BACKOFF = 2.0     # seconds
GROQ_MAX_BACKOFF = 60.0        # seconds
GROQ_CALL_TIMEOUT = 120        # seconds
GROQ_TOTAL_DEADLINE = 420      # seconds across all attempts

MISTRAL_MAX_ATTEMPTS = 3       # 1 initial request + 2 retries (429 / 5xx only)
MISTRAL_MAX_WAIT = 20.0        # seconds; cap on any single Mistral retry wait


def _backoff_seconds(attempt, retry_after=None):
    """Exponential backoff with jitter. Honours Groq's Retry-After header."""
    if retry_after is not None:
        try:
            base = max(0.0, float(retry_after))
        except (TypeError, ValueError):
            base = GROQ_INITIAL_BACKOFF * (2 ** attempt)
    else:
        base = GROQ_INITIAL_BACKOFF * (2 ** attempt)
    jitter = random.uniform(0.85, 1.15)
    return min(GROQ_MAX_BACKOFF, base * jitter)


def groq(model, prompt, max_tokens=900, json_mode=True,
         retries=GROQ_MAX_ATTEMPTS - 1, total_timeout=GROQ_TOTAL_DEADLINE):
    """Call Groq with resilient handling for transient 429/5xx/network errors.

    Permanent client errors (for example 401, 403, 404, and 422) are not
    retried. Rate-limit responses honour the API's Retry-After header when
    present; otherwise they use exponential backoff with jitter.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {GROQ}",
        "Content-Type": "application/json",
    }
    deadline = time.monotonic() + total_timeout
    last_error = None

    for attempt in range(retries + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=(10, min(GROQ_CALL_TIMEOUT, max(1, int(remaining)))),
            )
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < retries:
                wait = _backoff_seconds(attempt)
                print(f"  {model} network error, waiting {wait:.1f}s "
                      f"(attempt {attempt + 1}/{retries + 1}): {e}")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq {model} network error after "
                               f"{retries + 1} attempts: {e}") from e

        if r.status_code == 429:
            retry_after = r.headers.get("retry-after") or r.headers.get("Retry-After")
            if attempt < retries:
                wait = _backoff_seconds(attempt, retry_after)
                print(f"  {model} 429, waiting {wait:.1f}s "
                      f"(attempt {attempt + 1}/{retries + 1})")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"Groq {model} rate-limited after {retries + 1} attempts "
                f"(last Retry-After: {retry_after})"
            )

        if 500 <= r.status_code < 600:
            last_error = RuntimeError(
                f"Groq {model} HTTP {r.status_code}: {r.text[:200]}"
            )
            if attempt < retries:
                wait = _backoff_seconds(attempt)
                print(f"  {model} HTTP {r.status_code}, waiting {wait:.1f}s "
                      f"(attempt {attempt + 1}/{retries + 1})")
                time.sleep(wait)
                continue
            raise last_error

        if not r.ok:
            # Do not retry permanent request/model errors.
            raise RuntimeError(f"Groq {model} HTTP {r.status_code}: {r.text[:200]}")

        try:
            content = r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as e:
            last_error = e
            if attempt < retries:
                wait = _backoff_seconds(attempt)
                print(f"  {model} malformed response, waiting {wait:.1f}s "
                      f"(attempt {attempt + 1}/{retries + 1})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Groq {model} returned an unreadable response") from e

        return parse_json(content)

    raise RuntimeError(
        f"Groq {model} failed after {retries + 1} attempts: {last_error}"
    )


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
    if isinstance(result, (dict, list)):
        return result
    return parse_json(result)


def mistral(model, prompt, max_tokens=900, json_mode=True):
    """OpenAI-compatible call to Mistral's API. Used only as the independent
    fallback when Groq fails -- it must be a DIFFERENT model from both
    Cloudflare (researcher A) and Groq (researcher B) so the ceasefire
    indicator's two-model agreement check stays honest."""
    if not MISTRAL_API_KEY:
        raise RuntimeError("MISTRAL_API_KEY not set")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    # Mirror groq(): callers that expect a JSON array (e.g. the merge step)
    # pass json_mode=False, because json_object mode only allows an object.
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    # Short retry on rate limits (429) and server errors (5xx) only: a
    # per-second burst limit clears in a moment, anything else fails fast.
    r = None
    for attempt in range(MISTRAL_MAX_ATTEMPTS):
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {MISTRAL_API_KEY}",
                     "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        transient = (r.status_code == 429 or 500 <= r.status_code < 600)
        if transient and attempt < MISTRAL_MAX_ATTEMPTS - 1:
            wait = min(MISTRAL_MAX_WAIT,
                       _backoff_seconds(attempt, r.headers.get("Retry-After")))
            print(f"  mistral {model} HTTP {r.status_code}, waiting {wait:.1f}s "
                  f"(attempt {attempt + 1}/{MISTRAL_MAX_ATTEMPTS})")
            time.sleep(wait)
            continue
        break
    if not r.ok:
        raise RuntimeError(f"Mistral HTTP {r.status_code}: {r.text[:200]}")
    return parse_json(r.json()["choices"][0]["message"]["content"])


def groq_with_fallback(model, prompt, max_tokens=900, json_mode=True):
    """Call the requested Groq model, then use the configured free fallback.

    The fallback is for availability only: the returned result is still
    subjected to the same source, date, duplicate, confidence, and ceasefire
    validation as a primary-model result.
    """
    try:
        return groq(model, prompt, max_tokens=max_tokens, json_mode=json_mode)
    except Exception as e:
        print(f"  groq {model} failed: {e}; trying Mistral fallback "
              f"({FALLBACK_MODEL})")
        try:
            return mistral(
                FALLBACK_MODEL,
                prompt,
                max_tokens=max(1200, max_tokens),
                json_mode=json_mode,
            )
        except Exception as fallback_error:
            raise RuntimeError(
                f"Primary model {model} failed ({e}); Mistral fallback "
                f"also failed ({fallback_error})"
            ) from fallback_error

# ---------- prompts ----------
SCHEMA = """{
  "current": "2-4 sentences on the present situation",
  "sources": [{"label": "Outlet: headline", "url": "https://..."}],
  "events": [
    {"date": "YYYY-MM-DD", "date_quote": "verbatim snippet from an article showing this date",
 "time": "HH:MM or null", "timezone": "UTC or zone or local",
 "time_source_count": 0, "time_variance_minutes": 0,
 "title": "...", "description": "..."}
  ],
  "perspective": {
    "dominant_frame": "2-3 sentences or null",
    "divergence": "low" | "moderate" | "high" | null
  },
  "escalation": {
    "score": 0,
    "note": "one short sentence of evidence"
  },
  "ceasefire": {
    "status": "in_force" | "not_in_force" | "unclear" | "not_applicable",
    "evidence_ids": ["E1"]
  },
  "confidence": "high" | "medium" | "low"
}"""

TIME_RULES = """
DATE-QUOTE RULES (strict — apply to every event):
- "date_quote" MUST be the exact phrase from the article that states WHEN
  the event happened. Examples: "on Wednesday", "Thursday morning",
  "overnight", "on 17 November", "two weeks after the 2 November strike".
- The date you write in "date" MUST be resolvable from that quote plus the
  article's publication date. Do NOT use the article's publication date as
  the event date unless the article explicitly says the event occurred on
  the day of publication.
- If no phrase in the article states when the event happened, OMIT the
  event entirely. An omitted event is better than a misdated one.
- DO NOT guess. If two sources give different event dates, use the date
  that appears in at least two sources; note this in "description".

TIME-VERIFICATION RULES (strict):
- Extract the exact date and time the EVENT actually happened, NOT the date the article was published or updated.
- OVERNIGHT & PREVIOUS-EVENING RULE: Morning or next-day articles frequently report attacks or strikes that happened "overnight," "late last night," "previous night," or "yesterday evening." When a source dated on day D reports an overnight or prior-evening strike, set the event date to D - 1 (not D) unless the text explicitly places the strike after midnight on day D.
- RETROSPECTIVE / WRAP-UP RULE: If a recent article (e.g., published on Nov 16) discusses an event that occurred earlier (e.g., "two weeks after the Nov 2 strike" or referencing a past date), you MUST extract the original date the event happened (e.g., 2026-11-02). Never assign a recent report or summary date to an event that happened days or weeks prior.
- Cluster hour:minute values; 2+ sources within 25 min = valid cluster.
- "time" = MEDIAN HH:MM of largest cluster. None → null.
- "time_source_count" = size of cluster.
- "time_variance_minutes" = (max-min). Single source = 0. No cluster = 999.
- "timezone" = reported zone or "local".
- Never fabricate times.
"""

SOURCE_TIERS_PROMPT = (
    "SOURCE TIERS (editorial framework — apply the weighting rules below):\n"
    "Tier 1 (authoritative baseline — prefer for factual claims):\n"
    "  " + ", ".join(TIER_1_DOMAINS) + "\n"
    "Tier 2 (institutional / analysis — use for figures, context, legal framing):\n"
    "  " + ", ".join(TIER_2_DOMAINS) + "\n"
    "Tier 3 (regional specialists — use for perspective and on-the-ground detail):\n"
    "  " + ", ".join(TIER_3_DOMAINS) + "\n"
)

SOURCE_WEIGHTING_RULES = """
SOURCE-WEIGHTING RULES (strict — apply when the results disagree):
- Factual claims about what happened (actions, actors, dates, casualties)
  should be anchored in a Tier 1 source whenever one is available.
- For casualty and displacement figures, prefer Tier 1 or Tier 2 sources.
- Tier 3 sources are valuable for regional perspective and on-the-ground
  detail. When a Tier 3 source makes a factual claim that no Tier 1 source
  confirms, do not state it as fact — attribute it ("according to X") and
  lower your overall confidence by one level.
- If only Tier 3 sources support the entire "current" paragraph, set
  confidence to "low".
- Never average or blend conflicting figures from different tiers — cite
  the range and attribute each end to its source.
"""

PERSPECTIVE_RULES = """
PERSPECTIVE (2-3 sentences — describe framing, do NOT editorialise):
- "dominant_frame": 2-3 sentences describing how the sources in the results
  above are framing this conflict RIGHT NOW. Mention Tier 1 (Western wire /
  broadcaster) framing and Tier 3 (regional specialist) framing if both are
  present. If they agree, say so. If they diverge on what to emphasise or on
  who is responsible, describe both.
- "divergence": one of "low", "moderate", "high" — how much Tier 1 and Tier 3
  sources disagree on framing. Use "low" if all sources tell the same story,
  "moderate" if they emphasise different facts, "high" if they contradict
  each other on material points.
- Base this ONLY on the search results above. Do NOT add your own view of
  the conflict. Do NOT pick a side. Describe what is being said, not what
  is true.
- If the search results are too thin to compare framings, set "dominant_frame"
  to null and "divergence" to null.
"""

ESCALATION_RULES = """
ESCALATION SCORE (0-100 integer — be CONSERVATIVE and STABLE):
Assess how escalated this conflict is RIGHT NOW, relative to its own baseline.
Anchor the score to observable evidence, not rhetoric:

  0-20    De-escalated / frozen: ceasefire holding, active talks, no new violence
  21-40   Simmering: sporadic incidents, negotiations stalled but ongoing
  41-60   Elevated: recurring violence, failed talks, rising rhetoric
  61-80   High: sustained offensives, mass displacement, diplomacy collapsed
  81-100  Critical: full-scale war, regime collapse risk, mass-casualty events

STABILITY RULES (critical — this is the most important part):
- If the current situation is substantively UNCHANGED from the previous run,
  return the SAME score as the previous run provided below.
- Do NOT adjust the score because minor stories appeared in the news.
- Do NOT adjust the score based on your own analytical view. Anchor only
  to what the search results show about concrete events (attacks, talks,
  casualties, territorial changes, official statements).
- If the search results are thin or ambiguous, prefer the PREVIOUS SCORE
  over your own guess.
- Cite ONE short sentence of evidence in "note" — the specific event or
  fact driving the score. Keep it under 120 characters.
- The system will smooth your answer against the previous score before
  publishing, so do not try to compensate for past runs.
"""

EVENT_NEUTRALITY_RULES = """
EVENT NEUTRALITY & SAME-INCIDENT RULES:
- SAME-INCIDENT RULE: Never split opposing claims or different reporting
  angles of the SAME underlying incident into two separate events. Combine
  both sides into ONE single event entry.
- BOTH-SIDES ATTRIBUTION: In event descriptions involving contested military
  strikes, targets, casualties, interceptions, or responsibility claims,
  explicitly attribute claims ("X claimed", "Y reported", "according to Z")
  and include available counter-claims or denials from the search results
  rather than presenting contested assertions as unattributed fact.
"""

def research_prompt(conflict, results, recent_events, previous_escalation, cf_block=""):
    ctx = "\n".join(
        f"- {x['title']} — {x.get('content','')[:350]} ({x['url']})"
        for x in results
    )
    return f"""Today is {date.today().isoformat()}.
Conflict: {conflict['name']} (tier: {conflict['tier']}).

Verified search results from the last 10 days:
{ctx}

Events ALREADY recorded for this conflict (do NOT re-report these;
if new information refines an existing event, do NOT create a new event):
{recent_events}

Return ONLY valid JSON, exactly this shape:
{SCHEMA}

{TIME_RULES}

{SOURCE_TIERS_PROMPT}

{SOURCE_WEIGHTING_RULES}

{EVENT_NEUTRALITY_RULES}

{PERSPECTIVE_RULES}

{ESCALATION_RULES}

Previous escalation score (from last run): {previous_escalation if previous_escalation is not None else "none — first run for this conflict"}

{cf_block}

Other rules:
- Every claim in "current" must be supported by a result above.
- "sources": AT LEAST 3, up to 5. Only URLs from the list above.
- "events": 0-3 items, last 7 days only, appearing in 2+ results,
  not already recorded above. Each title must be a short unique label.
- Every string under 400 characters."""

def review_prompt(conflict, results, a, b, flags, recent_events, previous_escalation):
    ctx = "\n".join(
        f"- {x['title']} — {x.get('content','')[:350]} ({x['url']})"
        for x in results
    )
    return f"""Today is {date.today().isoformat()}.
REVIEWER for "{conflict['name']}". Two models disagreed.

Search results:
{ctx}

Model A:
{json.dumps(a, ensure_ascii=False)}

Model B:
{json.dumps(b, ensure_ascii=False)}

Flags: {', '.join(flags)}

Already recorded (do NOT re-report):
{recent_events}

{SOURCE_TIERS_PROMPT}

{SOURCE_WEIGHTING_RULES}

{EVENT_NEUTRALITY_RULES}

{PERSPECTIVE_RULES}

{ESCALATION_RULES}

Previous escalation score (from last run): {previous_escalation if previous_escalation is not None else "none"}

Output ONLY JSON in this shape:
{SCHEMA}

{TIME_RULES}

If results don't resolve the dispute, confidence = "low"."""

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
    with open("data/conflicts.json") as f:
        conflicts = json.load(f)
    with open("data/history.json") as f:
        history = json.load(f)

    merge_groups = find_same_day_groups(history)
    if merge_groups:
        print(f"Found {len(merge_groups)} same-day group(s) to review for merging")
        merges = merge_groups_with_llm(merge_groups, history)
        if merges:
            before = len(history)
            history = apply_merges(history, merges)
            print(f"Merged {before - len(history)} duplicate entr{'y' if before-len(history)==1 else 'ies'}")
        else:
            print("No merges applied from same-day groups "
                  "(check for 'merge LLM failed' lines above)")
    else:
        print("No same-day duplicate groups to merge")

    history, pruned = prune_history_duplicates(history)
    if pruned:
        print(f"Pruned {pruned} cross-day duplicate(s)")

    seen = {h["id"] for h in history}
    searches_per_run = len(conflicts)

    usage = load_usage()
    now = datetime.now(timezone.utc)
    is_second_run = (now.hour >= 12)
    advanced_count, info = compute_advanced_this_run(usage, now, is_second_run, searches_per_run)

    print(f"Tavily cycle {usage['cycle']}: {usage['credits_used']}/{usage['cycle_budget']} used")
    print(f"Run type: {'SECOND' if is_second_run else 'FIRST'}")
    print(f"Source tiers: {len(TIER_1_DOMAINS)} Tier 1 / {len(TIER_2_DOMAINS)} Tier 2 / {len(TIER_3_DOMAINS)} Tier 3")
    print(f"Plan: {advanced_count} advanced + {searches_per_run - advanced_count} basic "
          f"= {advanced_count*2 + (searches_per_run - advanced_count)} credits")
    print(f"Reason: {info['reason']}")
    print("")

    save_usage(usage)

    ok = 0
    skipped_dupes = 0
    confidence_adjustments = 0
    pending_events = []
    date_rejections = 0
    # Diagnostics counters (log only, no effect on the data written):
    logged_tavily_keys = False
    ev_reached_time_check = 0
    ev_model_time = 0
    ev_time_kept = 0
    for i, c in enumerate(conflicts):
        use_adv = (i < advanced_count)
        depth = "advanced" if use_adv else "basic"
        try:
            # Load previous escalation score (for stability anchoring)
            prev_escalation = None
            if isinstance(c.get('escalation'), dict):
                try:
                    prev_escalation = int(c['escalation'].get('score'))
                except (ValueError, TypeError):
                    prev_escalation = None

            results = tavily(f"{c['name']} conflict latest news", depth=depth)
            usage["credits_used"] += (2 if use_adv else 1)
            save_usage(usage)

            if not results:
                print(f"SKIP {c['name']}: no search results")
                continue

            tavily_urls = [x['url'] for x in results if x.get('url')]

            # --- Build map of URL -> publication date from Tavily metadata ---
            url_to_pub_date = {}
            for r in results:
                u = r.get('url')
                pd = r.get('published_date') or r.get('published') or ''
                if u and pd:
                    try:
                        dt = datetime.fromisoformat(pd.replace('Z', '+00:00'))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        url_to_pub_date[u.split('?')[0].rstrip('/')] = dt.date()
                    except (ValueError, TypeError):
                        try:
                            url_to_pub_date[u.split('?')[0].rstrip('/')] = date.fromisoformat(pd[:10])
                        except (ValueError, TypeError):
                            pass

            # Capture the earliest source publish time (when the story broke)
            src_time, src_meta = earliest_source_time_from_results(results)
            if src_time:
                c["latest_source_time"] = src_time.isoformat(timespec="seconds")
                if src_meta:
                    try:
                        host = urlparse(src_meta.get('url', '')).hostname or ''
                        if host.startswith('www.'):
                            host = host[4:]
                        if host:
                            c["latest_source_domain"] = host
                    except Exception:
                        pass

            # Diagnostics (log only, no behaviour change): why is
            # latest_source_time empty? Shows whether Tavily returns dates at all.
            pub_vals = [p for p in (r.get('published_date') or r.get('published')
                                    for r in results) if p]
            if not logged_tavily_keys and isinstance(results[0], dict):
                logged_tavily_keys = True
                print(f"  [diag] Tavily result fields: {sorted(results[0].keys())}")
            print(f"  [diag] {c['name']}: {len(pub_vals)}/{len(results)} Tavily results "
                  f"carry a published_date; usable={'yes' if src_time else 'no'}"
                  + ("" if (src_time or not pub_vals)
                     else f"; unparsed sample={pub_vals[0]!r}"))

            recent = recent_events_for_prompt(history, c['name'])

            # --- Ceasefire indicator evidence: sentences copied by code from the results above
            #     (+ optional free page downloads). No extra Tavily credits are used. ---
            cf_on = c.get("id") not in cfi.EXCLUDED_CONFLICTS
            cf_evid, cf_block = [], ""
            if cf_on:
                cf_texts = cfi.fetch_texts(results, tier_of) if os.environ.get("CEASEFIRE_FETCH", "1") == "1" else {}
                cf_evid = cfi.extract_evidence(results, cf_texts, datetime.now(timezone.utc).date(), tier_of)
                cf_block = cfi.prompt_block(cf_evid)
            prompt = research_prompt(c, results, recent, prev_escalation, cf_block)

            a = None
            try:
                a = cloudflare("@cf/meta/llama-3.3-70b-instruct-fp8-fast", prompt)
                time.sleep(1.0)
            except Exception as e:
                print(f"  cloudflare failed for {c['name']}: {e}")

            b = None
            try:
                b = groq_with_fallback(
                    "qwen/qwen3.8-27b", prompt, max_tokens=900
                )
                time.sleep(8.0)
            except Exception as e:
                print(f"  groq failed for {c['name']}: {e}")
                time.sleep(8.0)

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
                        final = groq_with_fallback(
                            "openai/gpt-oss-120b",
                            review_prompt(c, results, a, b, flags, recent, prev_escalation),
                            max_tokens=2000,
                            json_mode=False,
                        )
                        time.sleep(8.0)
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

            raw_confidence = final.get("confidence") or "medium"
            adjusted_conf, adjust_reason = adjust_confidence(raw_confidence, live)
            if adjust_reason:
                confidence_adjustments += 1
                print(f"  confidence adjusted for {c['name']}: "
                      f"{raw_confidence} → {adjusted_conf} ({adjust_reason})")

            c["current"]    = final["current"]
            c["sources"]    = live[:4]
            c["confidence"] = adjusted_conf
            c["verified_by"] = verified
            c["updated"]    = datetime.now(timezone.utc).isoformat(timespec="seconds")

            # --- Ceasefire indicator: green/red only when BOTH researchers agree and independent
            #     sources confirm it (see scripts/ceasefire_indicator.py); otherwise grey. ---
            if cf_on:
                run_now = datetime.now(timezone.utc)
                cf_a = a.get("ceasefire") if isinstance(a, dict) else None
                cf_b = b.get("ceasefire") if isinstance(b, dict) else None
                inds, cf_dec = cfi.build_indicator(c.get("indicators"), cf_a, cf_b, cf_evid,
                                                   run_now.isoformat(timespec="seconds"), run_now.date())
                c["indicators"] = inds
                cfi.log_decision("data/indicator_log.json", c["id"], cf_dec, cf_a, cf_b,
                                 run_now.isoformat(timespec="seconds"))
                print(f"  ceasefire indicator for {c['name']}: {cf_dec.state} ({cf_dec.reason or 'n/a'}, "
                      f"{len(cf_evid)} candidate sentences)")

            # --- Perspective (no smoothing, save as-is if model returned it) ---
            persp = final.get("perspective")
            if isinstance(persp, dict) and persp.get("dominant_frame"):
                c["perspective"] = {
                    "dominant_frame": str(persp["dominant_frame"])[:600],
                    "divergence": persp.get("divergence") or "low",
                }

            # --- Escalation (smooth against previous score) ---
            esc = final.get("escalation")
            raw_esc = None
            if isinstance(esc, dict):
                try:
                    raw_esc = int(esc.get("score"))
                except (ValueError, TypeError):
                    raw_esc = None
            if raw_esc is not None:
                smoothed = smooth_escalation(raw_esc, prev_escalation,
                                             adjusted_conf, len(live))
                if smoothed is not None:
                    delta_str = ""
                    if prev_escalation is not None:
                        delta = smoothed - prev_escalation
                        delta_str = f", Δ{delta:+d}"
                    print(f"  escalation for {c['name']}: {smoothed} "
                          f"(raw {raw_esc}, prev {prev_escalation if prev_escalation is not None else 'new'}{delta_str})")
                    c["escalation"] = {
                        "score": smoothed,
                        "note": (esc.get("note") or "")[:200],
                        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }

            ok += 1

            for ev in final.get("events", []):
                if not isinstance(ev, dict): continue
                if "date" not in ev or "title" not in ev: continue
                if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(ev.get("date", ""))):
                    continue

                run_date = datetime.now(timezone.utc).date()

                # --- Future-Proof Date Override Logic (runs FIRST) ---
                original_date = ev["date"]
                source_pub_date = None
                for s in live:
                    url_str = s.get("url", "")

                    # 1. ALWAYS check the URL path first (most reliable for true article date)
                    d_extracted = extract_date_from_url(url_str)
                    if d_extracted:
                        if 0 <= (run_date - d_extracted).days <= 14:
                            source_pub_date = d_extracted
                            break

                    # 2. Fallback to Tavily metadata only if URL has no date
                    clean_url = url_str.split('?')[0].rstrip('/')
                    if clean_url in url_to_pub_date:
                        d_meta = url_to_pub_date[clean_url]
                        if 0 <= (run_date - d_meta).days <= 14:
                            source_pub_date = d_meta
                            break

                # If LLM fell into recency bias (defaulting to today), correct it using an earlier trusted source date;
                # never push an earlier LLM-extracted event date forward onto a newer article publication date.
                if ev["date"] == run_date.isoformat() and source_pub_date and source_pub_date.isoformat() < ev["date"]:
                    print(f"  correcting event date via URL/metadata for {c['name']}: {ev['date']} → {source_pub_date.isoformat()}")
                    ev["date"] = source_pub_date.isoformat()

                # Track whether the URL override actually rescued the date
                url_rescued = (ev["date"] != original_date)

                # --- Guard 1 (pre-flight): reject far-future events BEFORE queueing ---
                # Runs before Guard 0 so impossible dates never consume external
                # verifier (scripts/verify_dates_free.py) queries.
                try:
                    ev_d_pre = date.fromisoformat(ev["date"])
                except (ValueError, TypeError):
                    continue

                # Guard 1: reject anything more than +1 day in the future
                # (allows legitimate timezone slips for Asia-Pacific timezones)
                if ev_d_pre > run_date + timedelta(days=1):
                    date_rejections += 1
                    print(f"  rejected far-future event for {c['name']}: "
                          f"{ev['title'][:50]} ({ev['date']})")
                    continue

                # --- Guard 0 (post-override): queue for external verification ---
                date_quote = str(ev.get("date_quote", "") or "").strip()
                if not url_rescued and (not date_quote or len(date_quote) < 4):
                    # Do NOT reject — queue for free-API date verification
                    # (scripts/verify_dates_free.py: DDG / GDELT / GNews).
                    pending_events.append({
                        "conflict": c["name"],
                        "tier": c["tier"],
                        "title": ev.get("title", ""),
                        "description": ev.get("description", ""),
                        "llm_date": ev["date"],
                        "llm_time": ev.get("time"),
                        "llm_timezone": ev.get("timezone"),
                        "queued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "status": "pending",
                    })
                    print(f"  queued for date verification: {c['name']} — {ev['title'][:60]}")
                    continue

                # --- Date guards (Guard 2 + 3; Guard 1 handled above) ---
                try:
                    ev_d = date.fromisoformat(ev["date"])
                except (ValueError, TypeError):
                    continue

                # Guard 2: reject if the source article was published BEFORE the
                # claimed event date (impossible — can't report on something that
                # hasn't happened). Tolerates ±1 day for timezone differences.
                if src_time and src_time.date() < (ev_d - timedelta(days=1)):
                    date_rejections += 1
                    print(f"  rejected impossible-date event for {c['name']}: "
                          f"{ev['title'][:50]} (event {ev['date']}, "
                          f"article published {src_time.date()})")
                    continue

                # Guard 3: reject stale events older than 14 days
                if (run_date - ev_d).days > 14:
                    date_rejections += 1
                    print(f"  rejected stale event for {c['name']}: "
                          f"{ev['title'][:50]} ({ev['date']})")
                    continue

                # Diagnostics (log only): what time information did the model give?
                ev_reached_time_check += 1
                if ev.get("time"):
                    ev_model_time += 1
                    print(f"  [diag] model time for {c['name']}: time={ev.get('time')!r} "
                          f"sources={ev.get('time_source_count')!r} "
                          f"variance={ev.get('time_variance_minutes')!r}")
                ev = normalize_event_time(ev)
                if ev.get("time") and ev.get("time_verified"):
                    ev_time_kept += 1
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

                if is_duplicate_event(candidate, history):
                    skipped_dupes += 1
                    print(f"  dup skipped for {c['name']}: {ev['title'][:60]}")
                    continue

                seen.add(eid)
                history.append(candidate)

            print(f"OK {c['name']} ({depth}, {len(live)} src, {c['confidence']})")

        except Exception as e:
            print(f"FAIL {c['name']}: {e}")

    history, pruned2 = prune_history_duplicates(history)
    if pruned2:
        print(f"Post-run cleanup: pruned {pruned2} additional duplicate(s)")

    history.sort(key=lambda h: h["date"], reverse=True)
    c_changed = save_if_changed("data/conflicts.json", conflicts)
    h_changed = save_if_changed("data/history.json",   history)
    if pending_events:
        pending_path = "data/pending_dates.json"
        try:
            with open(pending_path) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (FileNotFoundError, json.JSONDecodeError):
            existing = []

        # Fuzzy dedup: skip events already queued with a similar title in
        # the same conflict (Jaccard on title tokens >= 0.65). Catches
        # rephrased re-reports (e.g. "Iran attacks tanker" vs "Iran
        # attacks oil tanker") that would otherwise create duplicates.
        def _already_queued(p, existing_list):
            pt = _tokens(p.get("title", ""))
            for e in existing_list:
                if not isinstance(e, dict):
                    continue
                if e.get("conflict") != p["conflict"]:
                    continue
                et = _tokens(e.get("title", ""))
                if pt and et and _jaccard(pt, et) >= 0.65:
                    return True
            return False

        new_pending = [p for p in pending_events if not _already_queued(p, existing)]
        existing.extend(new_pending)
        with open(pending_path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Queued {len(new_pending)} new event(s) to data/pending_dates.json "
              f"for external verification ({len(pending_events) - len(new_pending)} already queued).")

    # Count total pending entries in the file (for summary footer)
    total_pending_after = 0
    if os.path.exists("data/pending_dates.json"):
        try:
            with open("data/pending_dates.json") as f:
                pd_data = json.load(f)
            if isinstance(pd_data, list):
                total_pending_after = len(pd_data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    print(f"\n{ok}/{len(conflicts)} refreshed; "
          f"conflicts {'updated' if c_changed else 'unchanged'}, "
          f"history {'updated' if h_changed else 'unchanged'}")
    print(f"Fuzzy-dedup skips: {skipped_dupes} new event(s) rejected as duplicates")
    print(f"Date-guard rejections: {date_rejections} event(s) rejected (future/impossible/stale)")
    print(f"Pending verification: {total_pending_after} event(s) in data/pending_dates.json")
    print(f"Confidence adjustments: {confidence_adjustments} conflict(s) re-graded by tier floor")
    print(f"Event times: {ev_model_time} of {ev_reached_time_check} event(s) reaching the time check "
          f"had a model-reported time; {ev_time_kept} passed the verification gate")
    print(f"Tavily total: {usage['credits_used']}/{usage['cycle_budget']} credits, "
          f"advanced days done: {usage['advanced_days_done']}")

if __name__ == "__main__":
    main()
