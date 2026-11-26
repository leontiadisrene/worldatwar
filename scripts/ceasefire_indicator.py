"""Ceasefire indicator: precision first, abstain when unsure.

Built ONLY from search results update.py already fetched (no extra Tavily credits) plus an
optional, free, best-effort download of the article pages themselves.

How a green or red dot is earned (all must hold, otherwise the dot is grey "unconfirmed"):
  1. CODE (not a model) copies ceasefire-related sentences out of the results and numbers them E1, E2...
     Models may only cite those ids, so no quote can be invented.
  2. BOTH researcher models (Cloudflare + Groq) give the same status and each cites at least one
     sentence whose wording, checked by regex in code, matches that status.
  3. The cited sentences come from at least 2 independent origins (a Reuters story reprinted on
     three sites counts once), at least one from a Tier 1/2 source, newest at most 4 days old,
     none older than 7 days, and all with a known publication date.
  4. No Tier 1/2 source (or two origins of any tier) states the opposite in the same results.
A previous green/red is carried forward for at most 10 days when a run simply finds no new
evidence (e.g. one model was rate-limited), but never when sources conflict or the models disagree.

BASELINE MODE (config/ceasefire_baseline.json, option A): a human-verified, sourced status per conflict
is the starting point and is HELD (up to `hold_days`, default 60) while nothing contradicts it:
  * the live strict rule above can confirm it (refreshes the date) or, if it says the opposite,
    start a pending flip that needs a second strict confirmation on a later run (>= 6 h, <= 3 days);
  * tripwires turn the row grey as soon as fresh reporting points the other way (see `tripwire`);
  * a standing status that is too old, or is being contested, is shown grey, never as fact.

Nothing here can guarantee a number like "98% sure". It is tuned for precision at the cost of recall,
and every decision is logged (data/indicator_log.json) so accuracy can be audited by sampling.
"""
from __future__ import annotations

import email.utils
import html
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from urllib.parse import urlparse

INDICATOR_ID = "ceasefire"
EXCLUDED_CONFLICTS = {"taiwan", "korea", "scs", "caucasus"}  # no active hostilities to cease: row hidden
MAX_AGE_DAYS = 7
NEWEST_MAX_AGE_DAYS = 4
CARRY_DAYS = 10
HOLD_DAYS = 60              # baseline mode: how long a standing status may be shown without re-confirmation
PENDING_MIN_HOURS = 6       # a flip must be confirmed by a different (later) run
PENDING_MAX_DAYS = 3
TRIP_RELEASE_DAYS = 7       # a tripped row stays grey this long after the last contradicting report
MAX_EVIDENCE = 16
MAX_PER_RESULT = 3

# ---------------------------------------------------------------- wording rules
_CF = r"(?:cease[\s-]?fire|truce)"
_KW = re.compile(r"cease[\s-]?fire|truce|armistice|peace (?:deal|agreement|accord)|hostilities", re.I)

_AUX = r"(?:(?:has|have|had|was|were|is|are|officially|formally|effectively|finally|abruptly|quietly|now|already)\s+){0,3}"
_QUAL = r"(?:\s+(?:between|in|with|for|over)\s+(?:\w+\s+){0,6}?\w+)?"
# explicit statements that no ceasefire exists / one was refused (a short duration does not weaken these)
_COLLAPSE_NONE = re.compile(
    r"\bno\s+(?:\w+\s+){0,2}" + _CF + r"\b(?!\s+(?:violation|breach|infraction|incident))"
    r"|\bwithout\s+(?:a\s+|any\s+)?" + _CF + r"\b"
    r"|\b(?:rejected|refused|ruled\s+out|dismissed|declined)\s+(?:\w+\s+){0,5}" + _CF + r"\b"
    r"|\b" + _CF + r"\s+(?:talks|negotiations|efforts)\s+(?:have\s+)?(?:stalled|failed|collapsed|broke\s+down)\b",
    re.I,
)
# a ceasefire that WAS in place and has ended (the ceasefire itself must be the subject: "the ceasefire that ended the war" does not match)
_COLLAPSE_END = re.compile(
    r"\b" + _CF + _QUAL + r"\s+" + _AUX + r"(?:collapsed|broke\s+down|broken\s+down|ended|expired|terminated|scrapped|abandoned|over|failed)\b"
    r"|\b(?:ended|scrapped|abandoned|withdrew\s+from|pulled\s+out\s+of|declared\s+(?:\w+\s+){0,2}(?:over|void))\s+(?:\w+\s+){0,3}" + _CF + r"\b"
    r"|\bresum\w+\s+(?:its\s+|the\s+)?(?:war|offensive|fighting|hostilities)\b",
    re.I,
)
_HOLDS = re.compile(
    _CF + r"\b[^.;]{0,50}\b(?:holds?|holding|held|in\s+(?:effect|force|place)|took\s+effect|came\s+into\s+(?:effect|force)|remains?|since)\b"
    r"|\b(?:despite|violat\w+|breach\w*|under|since|maintain\w*)\s+(?:\w+\s+){0,3}" + _CF + r"\b"
    r"|\b(?:signed|agreed|brokered|announced|reached)\s+(?:\w+\s+){0,3}" + _CF + r"\b",
    re.I,
)
# short, partial or holiday pauses never count as "a ceasefire in force"
_PARTIAL = re.compile(
    r"\b(?:(?:\d{1,2}|one|two|three|four|five|six|seven|eight)[\s-]*(?:hours?|days?)|holiday|easter|christmas|new\s+year|eid|ramadan|"
    r"temporary|brief|partial|limited|localis?ed|energy|humanitarian\s+(?:pause|window|corridor)|pause)\b", re.I)
# words that turn a statement into a hope, threat or condition rather than a fact
_SPECULATIVE = re.compile(
    r"\b(?:threaten\w*|warn\w*|vow\w*|could|would|might|(?-i:may)(?!\s+\d)|if|unless|risks?|fears?|feared|avert\w*|prevent\w*|hope\w*)\b", re.I)
_NEGATION = re.compile(
    r"(?:\b(?:no|not|never|fail\w*|without|propos\w*|plans?|planned|expected|expects?|aims?|seek\w*|calls?|called|urg\w*|demand\w*|talks|negotiat\w*|draft)\b|n't)",
    re.I)
_VIOL = re.compile(r"\b(?:violat\w+|breach\w*)\b|\bdespite\s+(?:\w+\s+){0,2}" + _CF + r"\b|\bsince\s+(?:\w+\s+){0,2}" + _CF + r"\b", re.I)

WIRES = {"reuters.com": "reuters", "apnews.com": "ap", "afp.com": "afp"}
_WIRE_TAG = re.compile(r"\((Reuters|AP|AFP)\)", re.I)
_SKIP_FETCH = ("reuters.com", "apnews.com", "nytimes.com", "washingtonpost.com", "bloomberg.com", "wsj.com", "ft.com")


def polarity(sentence: str) -> str | None:
    """'collapse' (no ceasefire / ended / rejected), 'holds' (ceasefire agreed and standing) or None."""
    if _SPECULATIVE.search(sentence):
        return None
    if _COLLAPSE_NONE.search(sentence):
        return "collapse"
    if _COLLAPSE_END.search(sentence) and not _PARTIAL.search(sentence):
        return "collapse"
    if _HOLDS.search(sentence) and not _NEGATION.search(sentence) and not _PARTIAL.search(sentence):
        return "holds"
    return None


# ---------------------------------------------------------------- helpers
def parse_date(v) -> date | None:
    if not v or not isinstance(v, str):
        return None
    try:
        dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date()
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date()
    except (TypeError, ValueError):
        pass
    try:
        return date.fromisoformat(v.strip()[:10])
    except ValueError:
        return None


def host_of(url: str) -> str:
    try:
        h = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return h[4:] if h.startswith("www.") else h


def origin_of(url: str, text: str) -> str:
    """Who produced the reporting: a named wire if syndicated, else the outlet's own label (bbc.co.uk == bbc.com)."""
    h = host_of(url)
    for dom, name in WIRES.items():
        if h == dom or h.endswith("." + dom):
            return name
    m = _WIRE_TAG.search((text or "")[:400])
    if m:
        return m.group(1).lower()
    parts = h.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "gov") and len(parts[-1]) == 2:
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else h


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", html.unescape(text or "")).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z“\"'])", text) if 40 <= len(s.strip()) <= 420]


# ---------------------------------------------------------------- free page text (best effort)
_fetch_clock = {"spent": 0.0}


def fetch_texts(results: list[dict], tier_fn, cap: int = 4, timeout: int = 6, budget_s: float = 240.0) -> dict[str, str]:
    """Download up to `cap` article pages (free, no API credits) to get past short search snippets.
    Skips sites that block bots, never raises, and stops once `budget_s` seconds were spent in total."""
    try:
        import requests
    except ImportError:  # pragma: no cover
        return {}
    out: dict[str, str] = {}
    order = sorted(results, key=lambda r: (tier_fn(r.get("url")) or 9))
    for r in order:
        if len(out) >= cap or _fetch_clock["spent"] >= budget_s:
            break
        url = r.get("url") or ""
        h = host_of(url)
        if not url or any(h == d or h.endswith("." + d) for d in _SKIP_FETCH):
            continue
        if any(_KW.search(s) for s in _sentences(r.get("content") or "")):
            continue  # the snippet already says something about it
        t0 = time.time()
        try:
            resp = requests.get(url, timeout=timeout, stream=True, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ConflictMap/1.0)", "Accept": "text/html,*/*"})
            ctype = resp.headers.get("Content-Type", "")
            if resp.status_code == 200 and "html" in ctype:
                raw = resp.raw.read(600_000, decode_content=True).decode(resp.encoding or "utf-8", "replace")
                raw = re.sub(r"(?is)<(script|style|noscript|nav|footer|header)\b.*?</\1>", " ", raw)
                paras = re.findall(r"(?is)<p\b[^>]*>(.*?)</p>", raw)
                text = " ".join(re.sub(r"(?s)<[^>]+>", " ", p) for p in paras)
                if len(text) > 400:
                    out[url] = html.unescape(text)[:20_000]
            resp.close()
        except Exception:
            pass
        _fetch_clock["spent"] += time.time() - t0
    return out


# ---------------------------------------------------------------- evidence extraction
def extract_evidence(results: list[dict], texts: dict[str, str], today: date, tier_fn, subject: list | None = None) -> list[dict]:
    """Ceasefire-related sentences from results that have a known date within MAX_AGE_DAYS.
    `subject` (optional): lowercase terms; a result is used only if its title, url or opening text names one,
    so a sentence about another conflict can never decide this one."""
    items, seen = [], set()
    terms = [t.lower() for t in (subject or []) if t]
    for r in results:
        url = r.get("url") or ""
        if terms:
            head = " ".join([r.get("title") or "", url, (r.get("content") or "")[:1500]]).lower()
            if not any(t in head for t in terms):
                continue
        pub = parse_date(r.get("published_date") or r.get("published") or "")
        if not url or pub is None:
            continue
        age = (today - pub).days
        if age > MAX_AGE_DAYS or age < -1:
            continue
        body = " ".join([texts.get(url, ""), r.get("content") or ""])
        origin = origin_of(url, body)
        cand = _sentences(r.get("title") or "") or ([r["title"].strip()] if r.get("title") else [])
        cand += _sentences(body)
        taken = 0
        for s in cand:
            key = re.sub(r"\W+", " ", s.lower())[:120]
            if not _KW.search(s) or key in seen:
                continue
            seen.add(key)
            items.append({"url": url, "domain": host_of(url), "origin": origin, "tier": tier_fn(url), "date": pub.isoformat(),
                          "age": age, "sentence": s, "polarity": polarity(s)})
            taken += 1
            if taken >= MAX_PER_RESULT:
                break
    items.sort(key=lambda e: ((e["tier"] or 9), e["age"]))
    items = items[:MAX_EVIDENCE]
    for i, e in enumerate(items, 1):
        e["id"] = f"E{i}"
    return items


PROMPT_RULES = """CEASEFIRE INDICATOR (strict; code will check your answer against the sentences below):
Return "ceasefire" in the JSON. "status" is one of:
- "in_force": a ceasefire, truce or armistice between the MAIN parties was agreed, has taken effect, and no sentence says it ended. Violations alone do not end it.
- "not_in_force": sentences state there is no ceasefire, one was rejected, collapsed, expired or ended, or only proposals/talks exist.
- "unclear": the sentences do not settle it, or they conflict. Use this whenever in doubt.
- "not_applicable": there are no active armed hostilities between organised parties.
A short or partial pause (a few days, one area, one type of target) is NOT "in_force".
"evidence_ids": the ids (like "E3") of the sentences below that support your status, copied exactly. Cite only ids from the list; never write a quote.
If the list is empty, answer "unclear"."""


def prompt_block(evid: list[dict]) -> str:
    if not evid:
        return PROMPT_RULES + "\n\nCEASEFIRE SENTENCES: (none found in the last 7 days)"
    lines = [f'{e["id"]} [{e["domain"]}, {e["date"]}] {e["sentence"]}' for e in evid]
    return PROMPT_RULES + "\n\nCEASEFIRE SENTENCES (copied from the search results by code):\n" + "\n".join(lines)


# ---------------------------------------------------------------- decision
@dataclass
class Decision:
    state: str                      # yes | no | unknown | hidden
    reason: str = ""                # conflict | disagree | no_evidence | single_model | ok
    evidence: list = field(default_factory=list)
    violations: bool = False


_STATUS = {"in_force", "not_in_force", "unclear", "not_applicable"}


def _status(cf) -> str | None:
    if not isinstance(cf, dict):
        return None
    s = str(cf.get("status", "")).strip().lower().replace("-", "_").replace(" ", "_")
    return s if s in _STATUS else None


def _cited(cf, by_id) -> list[dict]:
    out, seen = [], set()
    ids = (cf or {}).get("evidence_ids")
    for x in ids if isinstance(ids, list) else []:
        x = str(x).strip().upper()
        if x in by_id and x not in seen:
            seen.add(x)
            out.append(by_id[x])
    return out


def decide(a: dict | None, b: dict | None, evid: list[dict]) -> Decision:
    sa, sb = _status(a), _status(b)
    if sa is None or sb is None:
        return Decision("unknown", "single_model")
    if sa == sb == "not_applicable":
        return Decision("hidden")
    if sa != sb:
        return Decision("unknown", "disagree")
    if sa in ("unclear", "not_applicable"):
        return Decision("unknown", "no_evidence")
    want, opp = ("holds", "collapse") if sa == "in_force" else ("collapse", "holds")
    contra = [e for e in evid if e["polarity"] == opp]
    if any(e["tier"] in (1, 2) for e in contra) or len({e["origin"] for e in contra}) >= 2:
        return Decision("unknown", "conflict")
    by_id = {e["id"]: e for e in evid}
    ca = [e for e in _cited(a, by_id) if e["polarity"] == want]
    cb = [e for e in _cited(b, by_id) if e["polarity"] == want]
    if not ca or not cb:
        return Decision("unknown", "no_evidence")
    used = list({e["id"]: e for e in ca + cb}.values())
    if len({e["origin"] for e in used}) < 2 or not any(e["tier"] in (1, 2) for e in used):
        return Decision("unknown", "no_evidence")
    if min(e["age"] for e in used) > NEWEST_MAX_AGE_DAYS:
        return Decision("unknown", "no_evidence")
    shown, seen_origin = [], set()
    for e in sorted(used, key=lambda e: ((e["tier"] or 9), e["age"])):
        if e["origin"] not in seen_origin:
            seen_origin.add(e["origin"])
            shown.append({"outlet": e["domain"], "url": e["url"], "date": e["date"], "quote": e["sentence"]})
    viol = sa == "in_force" and any(_VIOL.search(e["sentence"]) for e in evid)
    return Decision("yes" if sa == "in_force" else "no", "ok", shown[:3], viol)


NOTES = {
    "conflict": "Reports conflict on this right now, so nothing is shown as fact.",
    "disagree": "The automated checks disagreed, so nothing is shown as fact.",
    "no_evidence": "Not enough recent, independently confirmed reporting to say.",
    "single_model": "Not enough recent, independently confirmed reporting to say.",
}


def _build_legacy(prev, d, now_iso: str, today: date):
    """No baseline for this conflict: strict live rule plus a short carry-forward."""
    if d.state == "hidden":
        return []
    base = {"id": INDICATOR_ID, "label": "Ceasefire in force", "checked_at": now_iso}
    if d.state in ("yes", "no"):
        return [{**base, "state": d.state, "violations": d.violations, "carried": False,
                 "last_confirmed": now_iso, "evidence": d.evidence, "note": ""}]
    prev_item = next((x for x in (prev or []) if isinstance(x, dict) and x.get("id") == INDICATOR_ID), None)
    if prev_item and d.reason in ("no_evidence", "single_model") and prev_item.get("state") in ("yes", "no"):
        lc = parse_date(prev_item.get("last_confirmed") or "")
        if lc and 0 <= (today - lc).days <= CARRY_DAYS:
            n = (today - lc).days
            return [{**prev_item, "checked_at": now_iso, "carried": True,
                     "note": f"Last confirmed {n} day{'s' if n != 1 else ''} ago; no new confirming reports this run."}]
    return [{**base, "state": "unknown", "violations": False, "carried": False,
             "last_confirmed": (prev_item or {}).get("last_confirmed"), "evidence": [],
             "note": NOTES.get(d.reason, NOTES["no_evidence"])}]


def load_baseline(path: str) -> dict:
    """{conflict_id: entry} from the maintainer-verified baseline file; {} if missing or broken (never raises)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    cons = data.get("conflicts") if isinstance(data, dict) else None
    return {k: v for k, v in cons.items() if isinstance(v, dict)} if isinstance(cons, dict) else {}


def tripwire(state: str, evid: list[dict]) -> bool:
    """True when fresh reporting points the other way from the standing status `state` ('yes'|'no').
    Deliberately easier to trip than the strict rule is to pass: a false trip only costs a grey dot."""
    opp = "collapse" if state == "yes" else "holds"
    fresh = [e for e in evid if e["polarity"] == opp and e["age"] <= NEWEST_MAX_AGE_DAYS]
    if not fresh:
        return False
    origins = {e["origin"] for e in fresh}
    if state == "yes":
        return len(origins) >= 2 or any(e["tier"] in (1, 2) and _COLLAPSE_END.search(e["sentence"]) for e in fresh)
    return len(origins) >= 2 and any(e["tier"] in (1, 2) for e in fresh)


def _hours_between(a_iso: str, b_iso: str) -> float | None:
    try:
        a = datetime.fromisoformat(str(a_iso).replace("Z", "+00:00"))
        b = datetime.fromisoformat(str(b_iso).replace("Z", "+00:00"))
        if a.tzinfo is None:
            a = a.replace(tzinfo=timezone.utc)
        if b.tzinfo is None:
            b = b.replace(tzinfo=timezone.utc)
        return (b - a).total_seconds() / 3600
    except ValueError:
        return None


def _standing(prev_item: dict | None, base: dict) -> dict | None:
    """The status currently 'held': the previous result, or the baseline when the maintainer set a newer
    (or, on the same day, a different) one. None when nothing yes/no is held."""
    held = None
    if prev_item:
        src = prev_item if prev_item.get("state") in ("yes", "no") else (prev_item.get("prior") or {})
        if src.get("state") in ("yes", "no"):
            held = {"state": src["state"], "last_confirmed": src.get("last_confirmed"), "evidence": src.get("evidence") or [],
                    "violations": bool(src.get("violations")), "basis": src.get("basis") or "live"}
    seed = None
    if base.get("status") in ("yes", "no") and parse_date(base.get("verified") or ""):
        seed = {"state": base["status"], "last_confirmed": base["verified"], "evidence": base.get("evidence") or [],
                "violations": bool(base.get("violations")), "basis": "baseline"}
    if held and seed:
        hl, sv = parse_date(held["last_confirmed"] or ""), parse_date(seed["last_confirmed"])
        if hl is None or sv > hl or (sv == hl and seed["state"] != held["state"]):
            return seed
        return held
    return held or seed


def _pack(st: dict) -> dict:
    return {k: st.get(k) for k in ("state", "last_confirmed", "evidence", "violations", "basis")}


def _build_baseline(prev, d: Decision, evid: list[dict], now_iso: str, today: date, base: dict):
    prev_item = next((x for x in (prev or []) if isinstance(x, dict) and x.get("id") == INDICATOR_ID), None)
    st = _standing(prev_item, base)
    try:
        hold = int(base.get("hold_days", HOLD_DAYS))
    except (TypeError, ValueError):
        hold = HOLD_DAYS
    head = {"id": INDICATOR_ID, "label": "Ceasefire in force", "checked_at": now_iso}
    word = {"yes": "in force", "no": "not in force"}

    def live(state):
        return {**head, "state": state, "violations": d.violations, "carried": False, "basis": "live",
                "last_confirmed": now_iso, "evidence": d.evidence, "note": ""}

    def grey(note, **extra):
        item = {**head, "state": "unknown", "violations": False, "carried": False, "basis": None,
                "last_confirmed": (st or {}).get("last_confirmed") or (prev_item or {}).get("last_confirmed"),
                "evidence": [], "note": note}
        item.update(extra)
        return [item]

    pend = (prev_item or {}).get("pending") or {}
    pend_age = _hours_between(pend.get("first_seen"), now_iso) if pend.get("first_seen") else None
    pend_alive = bool(pend) and pend_age is not None and 0 <= pend_age <= PENDING_MAX_DAYS * 24

    # 1. strict live verdict (both models + sources, see decide())
    if d.state in ("yes", "no"):
        if st is None or st["state"] == d.state:
            return [live(d.state)]
        if pend_alive and pend.get("state") == d.state and pend_age >= PENDING_MIN_HOURS:
            item = live(d.state)
            item["note"] = f"Changed from {word[st['state']]}; confirmed on two separate checks."
            return [item]
        first = pend["first_seen"] if pend_alive and pend.get("state") == d.state else now_iso
        return grey(f"Recent reports say this changed (was: {word[st['state']]}). Waiting for a second confirmation before showing it.",
                    prior=_pack(st), pending={"state": d.state, "first_seen": first, "evidence": d.evidence})

    # 2. no strict verdict this run
    if st is None:
        if base.get("status") == "hidden":
            return []
        return grey(base.get("note") or NOTES.get(d.reason, NOTES["no_evidence"]))
    lc = parse_date(st["last_confirmed"] or "")
    age = (today - lc).days if lc else None
    if age is None or age < 0 or age > hold:
        return grey(f"The last verified status ({word[st['state']]}) is {age if age is not None else 'many'} days old and was not re-confirmed.")
    if pend_alive:
        return grey(f"Recent reports say this changed (was: {word[st['state']]}). Waiting for a second confirmation before showing it.",
                    prior=_pack(st), pending=pend)
    trip = prev_item.get("tripped") if prev_item else None
    if tripwire(st["state"], evid):
        return grey(f"Fresh reports may contradict the last verified status ({word[st['state']]}, {st['last_confirmed'][:10]}). Shown as unconfirmed until clarified.",
                    prior=_pack(st), tripped={"first": (trip or {}).get("first") or now_iso, "last": now_iso})
    if trip:
        tl = parse_date(trip.get("last") or "")
        if tl and 0 <= (today - tl).days < TRIP_RELEASE_DAYS:
            return grey(f"Fresh reports may contradict the last verified status ({word[st['state']]}, {st['last_confirmed'][:10]}). Shown as unconfirmed until clarified.",
                        prior=_pack(st), tripped=trip)
    return [{**head, "state": st["state"], "violations": st["violations"], "carried": True, "basis": st["basis"],
             "last_confirmed": st["last_confirmed"], "evidence": st["evidence"],
             "note": f"Standing status, last verified {age} day{'s' if age != 1 else ''} ago; no contradicting reports."}]


def build_indicator(prev: list | None, a: dict | None, b: dict | None, evid: list[dict], now_iso: str, today: date,
                    baseline: dict | None = None):
    """Return (indicators_list, decision). The list is empty when the indicator does not apply.
    `baseline` is this conflict's entry from the baseline file; None keeps the original strict-only behaviour."""
    d = decide(a, b, evid)
    if baseline is None:
        if d.state == "hidden":
            return [], d
        return _build_legacy(prev, d, now_iso, today), d
    if d.state == "hidden":
        d = Decision("unknown", "no_evidence")      # both models said N/A: not enough on its own to override a human baseline
    return _build_baseline(prev, d, evid, now_iso, today, baseline), d


# ---------------------------------------------------------------- audit log
def log_decision(path: str, conflict_id: str, d: Decision, a, b, now_iso: str, keep: int = 1500, shown: list | None = None):
    try:
        try:
            with open(path, encoding="utf-8") as f:
                log = json.load(f)
            if not isinstance(log, list):
                log = []
        except (FileNotFoundError, json.JSONDecodeError):
            log = []
        log.append({"at": now_iso, "conflict": conflict_id, "state": d.state, "reason": d.reason,
                    "model_a": _status(a), "model_b": _status(b),
                    "shown": (shown[0].get("state") if shown else "hidden") if shown is not None else None,
                    "basis": (shown[0].get("basis") if shown else None) if shown is not None else None,
                    "evidence": [{"url": e["url"], "date": e["date"], "quote": e["quote"]} for e in d.evidence]})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log[-keep:], f, indent=1, ensure_ascii=False)
            f.write("\n")
    except OSError:
        pass
