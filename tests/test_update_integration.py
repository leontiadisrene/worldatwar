"""End-to-end test of scripts/update.py with fake search + fake models (no network, no keys).
Run: python -m unittest discover -s tests -v"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
for k in ("TAVILY_API_KEY", "GROQ_API_KEY", "CF_ACCOUNT_ID", "CF_API_TOKEN"):
    os.environ.setdefault(k, "test")
os.environ["CEASEFIRE_FETCH"] = "0"

import ceasefire_indicator as cfi  # noqa: E402
import update  # noqa: E402


def iso_days_ago(n):
    return (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")


def fake_results(cid):
    p = iso_days_ago(1)
    base = {
        "gaza": [
            ("https://www.reuters.com/world/gaza-a", "Strikes kill aid workers", "Israeli strikes killed three aid workers despite a ceasefire that took effect in October 2025, officials said."),
            ("https://www.bbc.com/news/gaza-b", "Gaza ceasefire strained", "The ceasefire has been in effect since October, but both sides accuse each other of violating the deal."),
            ("https://www.aljazeera.com/gaza-c", "Truce violations", "Hamas said Israel violated the truce again this week."),
        ],
        "ukraine": [
            ("https://www.reuters.com/world/ua-a", "Talks end", "Talks ended with a prisoner swap but there is no ceasefire in place, officials said."),
            ("https://www.theguardian.com/ua-b", "Kyiv struck", "Russia rejected a proposed ceasefire on Friday and launched fresh drone strikes on Kyiv overnight."),
            ("https://www.bbc.com/news/ua-c", "Front line", "Fighting continued along the front on Saturday."),
        ],
        "taiwan": [
            ("https://www.reuters.com/world/tw-a", "PLA drills", "China held drills near Taiwan."),
            ("https://www.bbc.com/news/tw-b", "Strait tensions", "Taipei protested new incursions."),
            ("https://www.theguardian.com/tw-c", "APEC", "Leaders met at APEC."),
        ],
    }[cid]
    return [{"url": u, "title": t, "content": c, "published_date": p} for u, t, c in base]


def fake_model_factory(mode):
    """Behaves like a careful researcher: cites only sentences it can see in the CEASEFIRE SENTENCES list."""
    def model(prompt, force_unclear=False):
        ids_hold, ids_collapse = [], []
        for m in re.finditer(r"^(E\d+) \[[^\]]*\] (.+)$", prompt, re.M):
            pol = cfi.polarity(m.group(2))
            (ids_hold if pol == "holds" else ids_collapse if pol == "collapse" else []).append(m.group(1))
        urls = re.findall(r"\((https://[^)\s]+)\)", prompt)[:3]
        if "CEASEFIRE SENTENCES" not in prompt:
            cf = None
        elif force_unclear:
            cf = {"status": "unclear", "evidence_ids": []}
        elif ids_hold and not ids_collapse:
            cf = {"status": "in_force", "evidence_ids": ids_hold}
        elif ids_collapse and not ids_hold:
            cf = {"status": "not_in_force", "evidence_ids": ids_collapse}
        else:
            cf = {"status": "unclear", "evidence_ids": []}
        out = {"current": "Test paragraph.", "sources": [{"label": "x", "url": u} for u in urls], "events": [],
               "perspective": {"dominant_frame": "x", "divergence": "low"},
               "escalation": {"score": 50, "note": "n"}, "confidence": "medium"}
        if cf:
            out["ceasefire"] = cf
        return out
    return model


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = os.getcwd()
        os.chdir(self.tmp)
        os.makedirs("data")
        conflicts = [{"id": i, "name": i.title(), "tier": t, "lat": 0, "lon": 0, "who": "", "what": "", "when": "", "how": "",
                      "current": "old", "sources": []} for i, t in (("gaza", "war"), ("ukraine", "war"), ("taiwan", "flash"))]
        with open("data/conflicts.json", "w") as f:
            json.dump(conflicts, f)
        with open("data/history.json", "w") as f:
            json.dump([], f)
        self.mode = {"cf_down": False}
        model = fake_model_factory(self.mode)
        update.tavily = lambda q, depth="basic": fake_results(next(i for i in ("gaza", "ukraine", "taiwan") if i in q.lower()))
        update.cloudflare = lambda m, prompt, max_tokens=1200: model(prompt)
        def groq(m, prompt, max_tokens=900, json_mode=True, retries=2, total_timeout=None):
            if self.mode["cf_down"]:
                raise RuntimeError("Groq 429")
            return model(prompt)
        update.groq = groq
        update.url_valid = lambda url, urls: True
        update.time.sleep = lambda s: None
        self._saved_gemini_chain = update.gemini_chain      # default: no Gemini key -> fallback unavailable
        update.gemini_chain = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("GEMINI_API_KEY not set"))

    def tearDown(self):
        update.gemini_chain = self._saved_gemini_chain
        os.chdir(self.old)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self):
        with open("data/conflicts.json") as f:
            return {c["id"]: c for c in json.load(f)}

    def test_full_run_then_partial_outage(self):
        update.main()
        c = self.load()
        g = c["gaza"]["indicators"][0]
        self.assertEqual((g["state"], g["carried"]), ("yes", False))
        self.assertEqual(c["gaza"]["verified_by"], "cloudflare+groq")      # provenance label unchanged on the normal path
        self.assertTrue(g["violations"])
        self.assertGreaterEqual(len({e["outlet"] for e in g["evidence"]}), 2)
        for e in g["evidence"]:
            self.assertTrue(e["url"].startswith("https://"))
            self.assertTrue(e["quote"])
        u = c["ukraine"]["indicators"][0]
        self.assertEqual(u["state"], "no")
        self.assertNotIn("indicators", c["taiwan"])          # excluded conflicts get no ceasefire row at all
        with open("data/indicator_log.json") as f:
            log = json.load(f)
        self.assertEqual({x["conflict"] for x in log}, {"gaza", "ukraine"})

        # second run: Groq is down. Previous green/red must be kept (marked carried), never flipped.
        self.mode["cf_down"] = True
        update.main()
        c2 = self.load()
        for cid, state in (("gaza", "yes"), ("ukraine", "no")):
            it = c2[cid]["indicators"][0]
            self.assertEqual((it["state"], it["carried"]), (state, True))
            self.assertIn("Last confirmed", it["note"])

    def test_baseline_keeps_dots_when_models_are_down(self):
        """The situation in the real Actions log: Cloudflare/Groq quota gone, no strict verdict possible."""
        today = datetime.now(timezone.utc).date().isoformat()
        ev = lambda o: [{"outlet": o, "date": today, "url": f"https://{o}/a", "quote": "q"}] * 2
        os.makedirs("config")
        json.dump({"conflicts": {
            "gaza": {"status": "yes", "violations": True, "verified": today, "subject": ["gaza"], "evidence": ev("a.com")},
            "ukraine": {"status": "no", "verified": today, "subject": ["ukrain"], "evidence": ev("b.com")},
        }}, open("config/ceasefire_baseline.json", "w"))
        self.mode["cf_down"] = True            # Groq (model B) down; model A answers alone -> single_model
        update.main()
        c = self.load()
        g, u = c["gaza"]["indicators"][0], c["ukraine"]["indicators"][0]
        self.assertEqual((g["state"], g["basis"], g["violations"]), ("yes", "baseline", True))
        self.assertEqual((u["state"], u["basis"]), ("no", "baseline"))
        self.assertNotIn("indicators", c["taiwan"])
        log = json.load(open("data/indicator_log.json"))
        self.assertTrue(all(x["shown"] in ("yes", "no") and x["basis"] == "baseline" for x in log))
        # a second run changes nothing and never flips
        update.main()
        c2 = self.load()
        self.assertEqual([c2[i]["indicators"][0]["state"] for i in ("gaza", "ukraine")], ["yes", "no"])

    def test_cloudflare_quota_gemini_replaces_A_and_label_is_honest(self):
        """Cloudflare out of free neurons: researcher A becomes Gemini, B stays Groq; the run still verifies with two
        different models and the published label names them."""
        update.cloudflare = lambda m, prompt, max_tokens=1200: (_ for _ in ()).throw(
            RuntimeError("Cloudflare HTTP 429: you have used up your daily free allocation of 10,000 neurons"))
        update.gemini_chain = lambda prompt, max_tokens=900, json_mode=True: fake_model_factory(self.mode)(prompt)
        update.main()
        c = self.load()
        self.assertEqual({c[i]["verified_by"] for i in ("gaza", "ukraine", "taiwan")}, {"gemini+groq"})
        self.assertEqual((c["gaza"]["indicators"][0]["state"], c["ukraine"]["indicators"][0]["state"]), ("yes", "no"))

    def test_cloudflare_and_groq_out_gives_gemini_plus_gptoss_never_gemini_twice(self):
        update.cloudflare = lambda m, prompt, max_tokens=1200: (_ for _ in ()).throw(RuntimeError("Cloudflare HTTP 429: daily free allocation"))
        model = fake_model_factory(self.mode)
        def groq(m, prompt, max_tokens=900, json_mode=True, retries=2, total_timeout=None):
            if "gpt-oss" in m:
                return model(prompt)
            raise RuntimeError("Groq 429")
        update.groq = groq
        update.gemini_chain = lambda prompt, max_tokens=900, json_mode=True: model(prompt)
        update.main()
        c = self.load()
        self.assertEqual({c[i]["verified_by"] for i in ("gaza", "ukraine", "taiwan")}, {"gemini+gpt-oss"})

    def test_everything_down_but_gemini_is_single_model_and_never_green(self):
        update.cloudflare = lambda m, prompt, max_tokens=1200: (_ for _ in ()).throw(RuntimeError("Cloudflare HTTP 429: daily free allocation"))
        update.groq = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Groq 429"))
        update.gemini_chain = lambda prompt, max_tokens=900, json_mode=True: fake_model_factory(self.mode)(prompt)
        update.main()
        c = self.load()
        self.assertEqual(c["gaza"]["verified_by"], "gemini-only")
        self.assertEqual(c["gaza"]["indicators"][0]["state"], "unknown")      # one model alone can never make a dot


if __name__ == "__main__":
    unittest.main()
