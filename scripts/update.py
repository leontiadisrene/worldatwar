#!/usr/bin/env python3
"""
Refresh conflict briefings using:
  - Tavily        (search, 1,000 credits/month free)
  - Cloudflare AI (Primary A: llama-3.3-70b, 10k Neurons/day free)
  - Groq          (Primary B: qwen3.8-27b, Reviewer: gpt-oss-120b, 1,000 RPD free)
"""
import json, os, time, hashlib, requests
from datetime import date, datetime, timezone

TAVILY    = os.environ["TAVILY_API_KEY"]
GROQ      = os.environ["GROQ_API_KEY"]
CF_ACCT   = os.environ["CF_ACCOUNT_ID"]
CF_TOKEN  = os.environ["CF_API_TOKEN"]

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

# ---------- tolerant JSON parse ----------
def parse_json(text):
    """Parse JSON, tolerating code fences and trailing content."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        raise ValueError(f"expected str or dict, got {type(text)}")
    s = text.strip()
    # strip markdown fences if present
    if s.startswith("```"):
        s = s.split("```", 2)
        s = s[1] if len(s) > 1 else s[0]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    # try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # try to trim trailing garbage after the last closing brace
        depth, end = 0, None
        for i, ch in enumerate(s):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: end = i + 1
        if end:
            return json.loads(s[:end])
        raise

# ---------- models ----------
def groq(model, prompt, max_tokens=2048):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }, timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Groq {model} HTTP {r.status_code}: {r.text[:200]}")
    content = r.json()["choices"][0]["message"]["content"]
    return parse_json(content)

def cloudflare(model, prompt):
    r = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCT}/ai/run/{model}",
        headers={"Authorization": f"Bearer {CF_TOKEN}", "Content-Type": "application/json"},
        json={"messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Cloudflare HTTP {r.status_code}: {r.text[:200]}")
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"CF error: {body.get('errors')}")
    result = body["result"]["response"]
    # Cloudflare sometimes returns a dict, sometimes a JSON string
    if isinstance(result, dict):
        return result
    return parse_json(result)

# ---------- prompts ----------
SCHEMA = """{
  "current": "2-4 sentences on the present situation",
  "sources": [{"label": "Outlet: headline", "url": "https://..."}],
  "events": [{"date": "YYYY-MM-DD", "title": "...", "description": "..."}],
  "confidence": "high" | "medium" | "low"
}"""

def research_prompt(conflict, results):
    ctx = "\n".join(
        f"- {x['title']} — {x.get('content','')[:350]} ({x['url']})"
        for x in results
    )
    return f"""Today is {date.today().isoformat()}.
Conflict: {conflict['name']} (tier: {conflict['tier']}).

Verified search results from the last 10 days:
{ctx}

Return ONLY valid JSON, no markdown, exactly this shape:
{SCHEMA}

Rules:
- Every factual claim in "current" must be supported by a result above.
- "sources": 3-5 items. Only URLs that appear verbatim in the results. Never invent URLs.
- "events": 0-3 items. Only include developments dated within the last 7 days that
  appear in at least two results. Otherwise return an empty list.
- Keep every string under 400 characters."""

def review_prompt(conflict, results, a, b, flags):
    ctx = "\n".join(
        f"- {x['title']} — {x.get('content','')[:350]} ({x['url']})"
        for x in results
    )
    return f"""Today is {date.today().isoformat()}.
You are the REVIEWER. Two models disagreed on the "{conflict['name']}" conflict.

Raw search results:
{ctx}

Model A output:
{json.dumps(a, ensure_ascii=False)}

Model B output:
{json.dumps(b, ensure_ascii=False)}

Disagreement flags: {', '.join(flags)}

Decide which claims are actually supported by the search results above. Repair
any unsupported claim. Return ONLY valid JSON in this exact shape:
{SCHEMA}

Rules:
- Base your answer strictly on the search results, not on either model's prose.
- Keep source URLs only if they appear verbatim in the results above.
- If the results genuinely do not resolve the dispute, set confidence to "low"."""

# ---------- structural comparison ----------
def compare(a, b):
    flags = []
    if a.get("confidence") != b.get("confidence"):
        flags.append("confidence mismatch")
    ua = {s.get("url","") for s in a.get("sources", []) if isinstance(s, dict)}
    ub = {s.get("url","") for s in b.get("sources", []) if isinstance(s, dict)}
    if ua != ub:
        flags.append("source-list mismatch")
    da = {e.get("date","") for e in a.get("events", []) if isinstance(e, dict)}
    db = {e.get("date","") for e in b.get("events", []) if isinstance(e, dict)}
    if da != db:
        flags.append("event-date mismatch")
    return flags

# ---------- validation ----------
def url_alive(url):
    try:
        h = requests.head(url, allow_redirects=True, timeout=8,
                          headers={"User-Agent": "Mozilla/5.0"})
        return h.status_code < 400
    except Exception:
        return False

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
    seen      = {h["id"] for h in history}
    ok        = 0

    for i, c in enumerate(conflicts):
        try:
            results = tavily(
                f"{c['name']} conflict latest news",
                depth="advanced" if i % 2 == 0 else "basic",
            )
            if not results:
                print(f"SKIP {c['name']}: no search results")
                continue

            prompt = research_prompt(c, results)

            a = None
            try:
                a = cloudflare("@cf/meta/llama-3.3-70b-instruct-fp8-fast", prompt)
                time.sleep(0.5)
            except Exception as e:
                print(f"  cloudflare failed for {c['name']}: {e}")

            b = None
            try:
                b = groq("qwen/qwen3.8-27b", prompt, max_tokens=2048)
                time.sleep(2.0)
            except Exception as e:
                print(f"  groq failed for {c['name']}: {e}")

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
                                     review_prompt(c, results, a, b, flags),
                                     max_tokens=2048)
                        time.sleep(2.0)
                        verified = "cloudflare+groq+groq-review"
                    except Exception as e:
                        print(f"  reviewer failed for {c['name']}: {e}; using A")
                        final, verified = a, "cloudflare+groq (no review)"
                else:
                    final, verified = a, "cloudflare+groq"

            # sanity check
            if not isinstance(final, dict) or "current" not in final:
                print(f"SKIP {c['name']}: final output missing 'current'")
                continue

            live = [s for s in final.get("sources", [])
                    if isinstance(s, dict) and s.get("url") and url_alive(s["url"])]
            if len(live) < 2:
                print(f"SKIP {c['name']}: fewer than 2 live sources")
                continue

            c["current"]    = final["current"]
            c["sources"]    = live[:5]
            c["confidence"] = final.get("confidence", "medium")
            c["verified_by"] = verified
            c["updated"]    = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ok += 1

            for ev in final.get("events", []):
                if not isinstance(ev, dict): continue
                if "date" not in ev or "title" not in ev: continue
                eid = hashlib.sha1(
                    f"{c['id']}|{ev['date']}|{ev['title']}".encode()
                ).hexdigest()[:12]
                if eid in seen:
                    continue
                seen.add(eid)
                history.append({
                    "id": eid, "date": ev["date"],
                    "conflict": c["name"], "tier": c["tier"],
                    "title": ev["title"], "description": ev.get("description",""),
                    "source": "auto",
                })

            print(f"OK {c['name']} ({len(live)} src, {final.get('confidence')})")

        except Exception as e:
            print(f"FAIL {c['name']}: {e}")

    history.sort(key=lambda h: h["date"], reverse=True)
    c_changed = save_if_changed("data/conflicts.json", conflicts)
    h_changed = save_if_changed("data/history.json",   history)
    print(f"\n{ok}/{len(conflicts)} refreshed; "
          f"conflicts {'updated' if c_changed else 'unchanged'}, "
          f"history {'updated' if h_changed else 'unchanged'}")

if __name__ == "__main__":
    main()
