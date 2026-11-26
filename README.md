# Where the World Is at War
      or WorldAtWar
An interactive map of fifteen active wars, high-tension flashpoints and
watchlist conflicts — each entry backed by cited wire reporting and a
visible confidence level.

**Live site:** https://leontiadisrene.github.io/worldatwar/

---

## What this is

A single-page, dependency-light conflict tracker. It shows:

- **Map tab** — every conflict plotted on a world map, colour-coded by
  severity (active war / flashpoint / watchlist). Tap a marker for a
  current briefing, the parties involved, key dates, and a full source
  list.
- **History tab** — a calendar of dated developments per conflict. Days
  with recorded events carry coloured dots; tap one to read what
  happened.

The page is static HTML + JavaScript. No framework, no build step, no
server. The data lives in two JSON files that a scheduled job refreshes
twice a day.

---

## How it stays current

A GitHub Actions workflow runs at **00:17 and 12:17 UTC** and executes
`scripts/update.py`. For each of the fifteen conflicts:

1. **Search** — queries Tavily, restricted to an allowlist of wire
   services and institutional trackers (AP, Reuters, AFP, BBC, Al
   Jazeera, The Guardian, NYT, Washington Post, France 24, ICG, UN, UNHCR,
   CFR, WHO, ICRC). Only reporting from the last 10 days is returned.
2. **Two independent researchers** — the *same* search results are sent
   to two models on two different providers:
   - Cloudflare Workers AI — `@cf/meta/llama-3.3-70b-instruct-fp8-fast`
   - Groq — `qwen/qwen3.8-27b`

   Both are asked for the same JSON schema: a short `current` paragraph,
   3–5 `sources` with verbatim URLs, any new dated `events`, and a
   `confidence` value.
3. **Structural comparison** — the two outputs are diffed on
   `confidence`, the set of source URLs, and the set of event dates. Any
   mismatch raises a flag.
4. **Conditional review** — only when a flag is raised does a third model
   run: Groq `openai/gpt-oss-120b`. It receives the raw search results
   plus both drafts and is instructed to decide which claims the
   evidence actually supports, repairing anything unsupported.
5. **Source gate** — every URL in the winning output is checked with a
   live HEAD request. Anything that doesn't resolve is dropped. A
   conflict left with fewer than two live citations is skipped entirely
   for that run, keeping its previous verified text rather than
   publishing something unverifiable.
6. **Calendar append** — new developments are added to
   `data/history.json`, deduplicated by a hash of
   `conflict + date + title`, so re-runs never duplicate.

The workflow commits the updated JSON files back to the repository. If
the site is served from GitHub Pages or Cloudflare Pages, that commit is
the deployment — no further action is needed.

---

## Reading the confidence flags

Every conflict panel shows a provenance line beneath its sources:

| `verified_by` value | What it means |
|---|---|
| `cloudflare+groq` | Both researchers agreed. Source gate passed. |
| `cloudflare+groq+groq-review` | Researchers disagreed; the reviewer resolved it against the raw evidence. |
| `gemini+groq`, `cloudflare+gemini`, `gemini+gpt-oss`, ... | Same as above, but one researcher was replaced by an independent fallback model because the usual one was out of free quota. The two researchers are always different model families. |
| `<model>-only` | Only one researcher answered (e.g. `groq-only`). |
| `Manually curated` | Seed entry, not yet refreshed by the pipeline. |

The `confidence` field comes from the model's own assessment of how
cleanly the search results supported its summary:

| Value | Interpretation |
|---|---|
| `high` | Multiple sources agree; no material ambiguity. |
| `medium` | Supported, but some details rest on single-source reporting. |
| `low` | The search results themselves were contradictory. Go read the citations. |

`low` is not an error state. It is the pipeline being honest about the
evidence.

---

## Repository layout

├── index.html
├── countries-110m.json
├── data/
│   ├── conflicts.json
│   └── history.json
├── scripts/
│   └── update.py
└── .github/
    └── workflows/
        └── update.yml
---

## Running locally

The page fetches its JSON at runtime, so opening `index.html` directly via
`file://` will fail — browsers block local `fetch()` for security reasons.
Serve it over HTTP instead:

```bash
python -m http.server 8080
# then open http://localhost:8080

Any static server works (npx serve, php -S, a phone app, etc.). The
topology file must sit next to index.html; the JSON files must be under
data/.

Adding a conflict

Append an object to data/conflicts.json:
{
  "id": "unique-slug",
  "name": "Display name",
  "lat": 0.0,
  "lon": 0.0,
  "tier": "war",
  "who": "Party A vs Party B",
  "what": "One-line description of the dispute",
  "when": "Key dates and current phase",
  "how": "How the fighting or standoff manifests",
  "current": "Seed paragraph; the refresh job will overwrite this.",
  "sources": [
    { "label": "Outlet: headline", "url": "https://..." }
  ]
}

tier must be one of war, flash, or watch — those map to the three
legend colours. The next scheduled refresh will start updating the new
entry automatically, as long as its name produces usable search results.

To add a historical milestone to the calendar, append to
data/history.json:

{
  "id": "unique-id",
  "date": "YYYY-MM-DD",
  "conflict": "Display name (must match conflicts.json)",
  "tier": "war",
  "title": "Short headline",
  "description": "One or two sentences.",
  "source": "archive"
}
