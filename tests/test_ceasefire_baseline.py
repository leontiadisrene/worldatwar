"""Option A: baseline hold, tripwires, two-run flips, expiry, subject guard. Run: python -m unittest discover -s tests -v"""
import json
import os
import sys
import tempfile
import unittest
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import ceasefire_indicator as cfi  # noqa: E402

T0 = "2026-11-26T01:00:00+00:00"
T1 = "2026-11-26T13:00:00+00:00"   # next run, 12 h later
TODAY = date(2026, 11, 26)


def ev(i, sentence, origin, tier=1, age=1, domain=None):
    return {"id": f"E{i}", "url": f"https://{domain or origin + '.com'}/a{i}", "domain": domain or origin + ".com", "origin": origin,
            "tier": tier, "date": "2026-11-25", "age": age, "sentence": sentence, "polarity": cfi.polarity(sentence)}


def base(status="yes", verified="2026-11-25", **kw):
    return {"status": status, "verified": verified, "violations": False,
            "evidence": [{"outlet": "x.com", "date": verified, "url": "https://x.com/a", "quote": "q"}], **kw}


HOLDS = [ev(1, "The ceasefire has been in effect since October, but both sides accuse each other of violations.", "reuters"),
         ev(2, "Hamas said Israel violated the truce again this week, despite the ceasefire.", "bbc", tier=1)]
NONE_ = [ev(1, "Talks ended with a prisoner swap but there is no ceasefire in place, officials said.", "reuters"),
         ev(2, "Russia rejected a proposed ceasefire on Friday and launched fresh drone strikes.", "guardian", tier=1)]
YES = {"status": "in_force", "evidence_ids": ["E1", "E2"]}
NO = {"status": "not_in_force", "evidence_ids": ["E1", "E2"]}


def run(prev, a, b, evid, now=T0, entry=None):
    inds, d = cfi.build_indicator(prev, a, b, evid, now, cfi.parse_date(now), baseline=entry)
    return inds, d


class Baseline(unittest.TestCase):
    def test_seed_is_shown_without_any_live_evidence(self):
        inds, _ = run(None, None, None, [], entry=base("yes"))
        it = inds[0]
        self.assertEqual((it["state"], it["basis"], it["carried"]), ("yes", "baseline", True))
        self.assertEqual(it["last_confirmed"], "2026-11-25")
        self.assertIn("1 day ago", it["note"])
        self.assertTrue(it["evidence"])

    def test_unknown_and_hidden_baselines(self):
        inds, _ = run(None, None, None, [], entry=base("unknown", note="Informal standoff."))
        self.assertEqual((inds[0]["state"], inds[0]["note"]), ("unknown", "Informal standoff."))
        self.assertEqual(run(None, None, None, [], entry=base("hidden"))[0], [])
        # both models saying not_applicable must not hide a human-verified row
        inds, _ = run(None, {"status": "not_applicable"}, {"status": "not_applicable"}, [], entry=base("yes"))
        self.assertEqual(inds[0]["state"], "yes")

    def test_models_down_or_disagreeing_do_not_disturb_the_held_status(self):
        for a, b in ((None, None), (None, YES), ({"status": "unclear"}, YES), ({"status": "not_in_force"}, YES)):
            inds, _ = run(None, a, b, [], entry=base("no"))
            self.assertEqual(inds[0]["state"], "no", (a, b))

    def test_live_confirmation_refreshes_and_replaces_baseline_evidence(self):
        inds, d = run(None, YES, YES, HOLDS, entry=base("yes"))
        it = inds[0]
        self.assertEqual((d.state, it["state"], it["basis"], it["carried"]), ("yes", "yes", "live", False))
        self.assertEqual(it["last_confirmed"], T0)

    def test_expiry_turns_grey(self):
        inds, _ = run(None, None, None, [], entry=base("yes", verified="2026-09-01"))
        self.assertEqual(inds[0]["state"], "unknown")
        self.assertIn("not re-confirmed", inds[0]["note"])
        inds, _ = run(None, None, None, [], entry=base("yes", verified="2026-10-20", hold_days=30))
        self.assertEqual(inds[0]["state"], "unknown")
        inds, _ = run(None, None, None, [], entry=base("yes", verified="2026-10-20", hold_days=60))
        self.assertEqual(inds[0]["state"], "yes")

    def test_flip_needs_two_separate_strict_runs(self):
        entry = base("yes")
        i1, d1 = run(None, NO, NO, NONE_, now=T0, entry=entry)            # strict opposite, first sighting
        self.assertEqual(d1.state, "no")
        self.assertEqual(i1[0]["state"], "unknown")
        self.assertEqual(i1[0]["pending"]["state"], "no")
        self.assertEqual(i1[0]["prior"]["state"], "yes")
        i1b, _ = run(i1, None, None, [], now=T0, entry=entry)             # same instant / no evidence: stays grey
        self.assertEqual(i1b[0]["state"], "unknown")
        i1c, _ = run(i1, NO, NO, NONE_, now="2026-11-26T02:00:00+00:00", entry=entry)   # < 6 h later: not a separate run
        self.assertEqual(i1c[0]["state"], "unknown")
        i2, _ = run(i1, NO, NO, NONE_, now=T1, entry=entry)               # 12 h later, confirmed again
        self.assertEqual((i2[0]["state"], i2[0]["basis"]), ("no", "live"))
        self.assertIn("two separate checks", i2[0]["note"])

    def test_pending_flip_expires_and_standing_returns(self):
        entry = base("yes")
        i1, _ = run(None, NO, NO, NONE_, now="2026-11-20T01:00:00+00:00", entry=entry)
        self.assertIn("pending", i1[0])
        inds, _ = run(i1, None, None, [], now=T0, entry=entry)            # 6 days later, never re-confirmed
        self.assertEqual(inds[0]["state"], "yes")

    def test_tripwire_yes_needs_two_origins_or_explicit_end_from_tier12(self):
        one_rejected = [ev(1, "Israel rejected a Hamas proposal for a ceasefire extension on Monday.", "reuters")]
        self.assertFalse(cfi.tripwire("yes", one_rejected))               # one origin, not an explicit "ended"
        two = one_rejected + [ev(2, "The government said there is no ceasefire with the group.", "bbc")]
        self.assertTrue(cfi.tripwire("yes", two))
        ended = [ev(1, "The ceasefire collapsed on Tuesday after heavy shelling, officials said.", "reuters")]
        self.assertTrue(cfi.tripwire("yes", ended))
        ended3 = [ev(1, "The ceasefire collapsed on Tuesday after heavy shelling, officials said.", "blog", tier=3)]
        self.assertFalse(cfi.tripwire("yes", ended3))
        stale = [ev(1, "The ceasefire collapsed on Tuesday after heavy shelling, officials said.", "reuters", age=6)]
        self.assertFalse(cfi.tripwire("yes", stale))

    def test_tripwire_no_needs_two_origins_with_tier12(self):
        a = ev(1, "The ceasefire has been in effect since Sunday and is holding, officials said.", "reuters")
        b = ev(2, "A ceasefire took effect on Sunday under the deal brokered by Qatar.", "bbc")
        self.assertTrue(cfi.tripwire("no", [a, b]))
        self.assertFalse(cfi.tripwire("no", [a]))
        self.assertFalse(cfi.tripwire("no", [ev(3, a["sentence"], "blog1", tier=3), ev(4, b["sentence"], "blog2", tier=3)]))

    def test_trip_turns_grey_then_stays_grey_a_week_then_returns(self):
        entry = base("yes")
        ended = [ev(1, "The ceasefire collapsed on Tuesday after heavy shelling, officials said.", "reuters")]
        i1, _ = run(None, None, None, ended, now=T0, entry=entry)
        self.assertEqual(i1[0]["state"], "unknown")
        self.assertIn("contradict", i1[0]["note"])
        i2, _ = run(i1, None, None, [], now="2026-11-28T01:00:00+00:00", entry=entry)   # quiet, but only 2 days later
        self.assertEqual(i2[0]["state"], "unknown")
        i3, _ = run(i1, None, None, [], now="2026-12-04T01:00:00+00:00", entry=entry)
        self.assertEqual(i3[0]["state"], "yes")

    def test_strict_confirmation_clears_a_trip_immediately(self):
        entry = base("yes")
        ended = [ev(1, "The ceasefire collapsed on Tuesday after heavy shelling, officials said.", "reuters")]
        i1, _ = run(None, None, None, ended, now=T0, entry=entry)
        i2, _ = run(i1, YES, YES, HOLDS, now=T1, entry=entry)
        self.assertEqual((i2[0]["state"], i2[0]["basis"]), ("yes", "live"))

    def test_newer_baseline_overrides_stored_result(self):
        i1, _ = run(None, None, None, [], entry=base("yes", verified="2026-11-20"))
        inds, _ = run(i1, None, None, [], entry=base("no", verified="2026-11-26"))
        self.assertEqual((inds[0]["state"], inds[0]["basis"]), ("no", "baseline"))
        inds, _ = run(i1, None, None, [], entry=base("no", verified="2026-11-20"))   # same day, different -> human wins
        self.assertEqual(inds[0]["state"], "no")

    def test_no_baseline_entry_keeps_strict_only_behaviour(self):
        inds, d = run(None, None, None, [], entry=None)
        self.assertEqual((inds[0]["state"], d.reason), ("unknown", "single_model"))
        inds, d = run(None, {"status": "not_applicable"}, {"status": "not_applicable"}, [], entry=None)
        self.assertEqual(inds, [])


class Loading(unittest.TestCase):
    def test_load_baseline_is_forgiving(self):
        self.assertEqual(cfi.load_baseline("/nonexistent/x.json"), {})
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not json")
        self.assertEqual(cfi.load_baseline(f.name), {})
        os.unlink(f.name)

    def test_shipped_baseline_is_valid(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "ceasefire_baseline.json")
        b = cfi.load_baseline(path)
        self.assertTrue(b)
        for cid, e in b.items():
            self.assertIn(e["status"], ("yes", "no", "unknown", "hidden"), cid)
            self.assertIsNotNone(cfi.parse_date(e["verified"]), cid)
            self.assertNotIn(cid, cfi.EXCLUDED_CONFLICTS)
            if e["status"] in ("yes", "no"):
                self.assertGreaterEqual(len(e["evidence"]), 2, cid)
                self.assertTrue(e["subject"], cid)
                for x in e["evidence"]:
                    self.assertTrue(x["url"].startswith("https://") and x["quote"] and cfi.parse_date(x["date"]), cid)


class SubjectGuard(unittest.TestCase):
    def test_results_about_other_conflicts_are_ignored(self):
        res = [{"url": "https://www.reuters.com/world/a", "title": "Russia rejects ceasefire in Ukraine", "published_date": "2026-11-25T10:00:00Z",
                "content": "Moscow said there is no ceasefire in sight, officials said on Tuesday."},
               {"url": "https://www.bbc.com/news/b", "title": "Gaza ceasefire strained", "published_date": "2026-11-25T10:00:00Z",
                "content": "The ceasefire in Gaza has held since October, but both sides accuse each other of violations."}]
        tier = lambda u: 1
        all_ = cfi.extract_evidence(res, {}, TODAY, tier)
        gaza = cfi.extract_evidence(res, {}, TODAY, tier, subject=["gaza", "hamas"])
        self.assertTrue(any("Ukraine" in e["sentence"] or "Moscow" in e["sentence"] for e in all_))
        self.assertTrue(gaza and all("Moscow" not in e["sentence"] for e in gaza))


if __name__ == "__main__":
    unittest.main()
