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

# ---------- models ----------
def groq(model, prompt, max_tokens=700, json_mode=True, retries=3):
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
            print(f"  {model} 429 (used {r.text.split('Used ')[1].split(',')[0] if 'Used ' in r.text else '?'}), waiting 20s to retry")
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
- "sources": include AT LEAST 3 items, 3-5 total. Only URLs from the list above. Never invent URLs.
- "events": 0-3 items. Only include developments dated within the last 7 days that
  appear in at least two results. Otherwise return an empty list.
- Keep every string under 400 characters."""

def review_prompt(conflict, results, a, b, flags):
    ctx = "\n".join(
        f"- {x['title']} — {x.get('content','')[:350]} ({x['url']})"
        for x in results
    )
    return f"""Today is {date.today().isoformat()}.
You are the REVIEWER for the "{conflict['name']}" conflict. Two models disagreed.

Search results:
{ctx}

Model A:
{json.dumps(a, ensure_ascii=False)}

Model B:
{json.dumps(b, ensure_ascii=False)}

Disagreement flags: {', '.join(flags)}

Decide which claims the search results actually support. Output ONLY the JSON
object, with no prose before or after, in this exact shape:
{SCHEMA}

If the results do not resolve the dispute, set confidence to "low"."""

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

            tavily_urls = [x['url'] for x in results if x.get('url')]
            prompt = research_prompt(c, results)

            a = None
            try:
                a = cloudflare("@cf/meta/llama-3.3-70b-instruct-fp8-fast", prompt)
                time.sleep(1.0)
            except Exception as e:
                print(f"  cloudflare failed for {c['name']}: {e}")

            b = None
            try:
                b = groq("qwen/qwen3.8-27b", prompt, max_tokens=700, retries=1)
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
                                     review_prompt(c, results, a, b, flags),
                                     max_tokens=2000, json_mode=False, retries=1)
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

            print(f"OK {c['name']} ({len(live)} src, {c['confidence']})")

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
