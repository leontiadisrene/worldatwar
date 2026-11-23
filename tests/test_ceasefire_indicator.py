"""Offline tests for scripts/ceasefire_indicator.py. Run: python -m unittest discover -s tests -v"""
import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import ceasefire_indicator as cf  # noqa: E402

TODAY = date(2026, 11, 23)
NOW = "2026-11-23T00:17:05Z"
TIERS = {"reuters.com": 1, "apnews.com": 1, "bbc.com": 1, "bbc.co.uk": 1, "theguardian.com": 1, "aljazeera.com": 3,
         "crisisgroup.org": 2, "timesofisrael.com": 3}


def tier(url):
    h = cf.host_of(url or "")
    for d, t in TIERS.items():
        if h == d or h.endswith("." + d):
            return t
    return None


def res(url, title, content, pub="2026-11-22"):
    return {"url": url, "title": title, "content": content, "published_date": pub}


GAZA = [
    res("https://www.reuters.com/world/gaza-1", "Israeli strikes kill aid workers in Gaza",
        "Israeli strikes killed three aid workers on Saturday despite a ceasefire that took effect in October 2025, Gaza officials said."),
    res("https://www.bbc.com/news/gaza-2", "Gaza ceasefire under strain",
        "The ceasefire has been in effect since October, but both sides accuse each other of violating the deal."),
    res("https://www.aljazeera.com/news/gaza-3", "Hamas says Israel violated truce",
        "Hamas said Israel repeatedly violated the truce this week. Mediators urged restraint."),
]
UKRAINE = [
    res("https://www.reuters.com/world/ua-1", "Talks end without ceasefire", "Talks ended with a prisoner swap but there is no ceasefire in place, Reuters reported."),
    res("https://www.theguardian.com/ua-2", "Kyiv strike", "Russia rejected a proposed ceasefire on Friday and launched fresh drone strikes on Kyiv overnight."),
]


def run(results, a, b, texts=None):
    ev = cf.extract_evidence(results, texts or {}, TODAY, tier)
    return ev, cf.decide(a, b, ev)


def ids_for(ev, pol, *domains):
    return [e["id"] for e in ev if e["polarity"] == pol and (not domains or e["domain"].endswith(domains))]


class Polarity(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(cf.polarity("Strikes killed three aid workers despite a ceasefire that took effect in October."), "holds")
        self.assertEqual(cf.polarity("Talks ended and there is no ceasefire in place."), "collapse")
        self.assertEqual(cf.polarity("Russia rejected a proposed ceasefire on Friday."), "collapse")

    def test_hedged_or_speculative_is_ignored(self):
        self.assertIsNone(cf.polarity("Israel threatened to resume the war if Hamas does not disarm."))
        self.assertIsNone(cf.polarity("Mediators proposed a ceasefire that would take effect next week."))
        self.assertIsNone(cf.polarity("Leaders called for a ceasefire since the fighting began."))
        self.assertIsNone(cf.polarity("Analysts fear the ceasefire could collapse."))


class Decisions(unittest.TestCase):
    def test_green_when_two_models_and_two_origins_agree(self):
        ev = cf.extract_evidence(GAZA, {}, TODAY, tier)
        ids = ids_for(ev, "holds")
        self.assertGreaterEqual(len({e["origin"] for e in ev if e["id"] in ids}), 2)
        d = cf.decide({"status": "in_force", "evidence_ids": ids[:1]}, {"status": "in_force", "evidence_ids": ids[1:2] or ids}, ev)
        self.assertEqual(d.state, "yes")
        self.assertTrue(d.violations)
        self.assertLessEqual(len(d.evidence), 3)

    def test_red_when_explicit_no_ceasefire(self):
        ev = cf.extract_evidence(UKRAINE, {}, TODAY, tier)
        ids = ids_for(ev, "collapse")
        d = cf.decide({"status": "not_in_force", "evidence_ids": ids}, {"status": "not_in_force", "evidence_ids": ids}, ev)
        self.assertEqual(d.state, "no")
        self.assertEqual({e["outlet"] for e in d.evidence}, {"reuters.com", "theguardian.com"})

    def test_models_disagree_is_grey(self):
        ev = cf.extract_evidence(GAZA, {}, TODAY, tier)
        ids = ids_for(ev, "holds")
        d = cf.decide({"status": "in_force", "evidence_ids": ids}, {"status": "unclear", "evidence_ids": []}, ev)
        self.assertEqual((d.state, d.reason), ("unknown", "disagree"))

    def test_single_origin_is_grey(self):
        ev = cf.extract_evidence(GAZA[:1], {}, TODAY, tier)
        ids = ids_for(ev, "holds")
        d = cf.decide({"status": "in_force", "evidence_ids": ids}, {"status": "in_force", "evidence_ids": ids}, ev)
        self.assertEqual(d.state, "unknown")

    def test_syndicated_wire_counts_once(self):
        rs = [res("https://www.reuters.com/a", "Ceasefire holds", "A ceasefire took effect last month and remains in place, officials said."),
              res("https://www.bbc.com/b", "Ceasefire holds", "(Reuters) - A ceasefire took effect last month and is in effect, officials said.")]
        ev = cf.extract_evidence(rs, {}, TODAY, tier)
        self.assertEqual({e["origin"] for e in ev}, {"reuters"})
        ids = [e["id"] for e in ev if e["polarity"] == "holds"]
        d = cf.decide({"status": "in_force", "evidence_ids": ids}, {"status": "in_force", "evidence_ids": ids}, ev)
        self.assertEqual(d.state, "unknown")

    def test_collapse_from_tier1_blocks_green(self):
        rs = GAZA + [res("https://www.apnews.com/x", "Truce over", "Israel said the ceasefire collapsed on Sunday after rocket fire.", "2026-11-23")]
        ev = cf.extract_evidence(rs, {}, TODAY, tier)
        ids = ids_for(ev, "holds")
        d = cf.decide({"status": "in_force", "evidence_ids": ids}, {"status": "in_force", "evidence_ids": ids}, ev)
        self.assertEqual((d.state, d.reason), ("unknown", "conflict"))

    def test_invented_or_wrong_ids_do_not_count(self):
        ev = cf.extract_evidence(GAZA, {}, TODAY, tier)
        d = cf.decide({"status": "in_force", "evidence_ids": ["E99"]}, {"status": "in_force", "evidence_ids": ["E98"]}, ev)
        self.assertEqual(d.state, "unknown")

    def test_ids_with_wrong_polarity_do_not_count(self):
        ev = cf.extract_evidence(UKRAINE, {}, TODAY, tier)
        ids = ids_for(ev, "collapse")
        d = cf.decide({"status": "in_force", "evidence_ids": ids}, {"status": "in_force", "evidence_ids": ids}, ev)
        self.assertEqual(d.state, "unknown")

    def test_stale_or_undated_results_are_not_evidence(self):
        old = [dict(r, published_date="2026-10-01") for r in UKRAINE]
        undated = [dict(r, published_date="") for r in UKRAINE]
        self.assertEqual(cf.extract_evidence(old, {}, TODAY, tier), [])
        self.assertEqual(cf.extract_evidence(undated, {}, TODAY, tier), [])

    def test_newest_evidence_must_be_recent(self):
        rs = [dict(r, published_date="2026-11-17") for r in UKRAINE]  # 6 days old: counted as evidence but too old to publish
        ev = cf.extract_evidence(rs, {}, TODAY, tier)
        ids = ids_for(ev, "collapse")
        d = cf.decide({"status": "not_in_force", "evidence_ids": ids}, {"status": "not_in_force", "evidence_ids": ids}, ev)
        self.assertEqual(d.state, "unknown")

    def test_not_applicable_hides_row(self):
        self.assertEqual(cf.decide({"status": "not_applicable"}, {"status": "not_applicable"}, []).state, "hidden")
        self.assertEqual(cf.build_indicator(None, {"status": "not_applicable"}, {"status": "not_applicable"}, [], NOW, TODAY)[0], [])

    def test_missing_model_is_not_enough(self):
        ev = cf.extract_evidence(GAZA, {}, TODAY, tier)
        ids = ids_for(ev, "holds")
        self.assertEqual(cf.decide({"status": "in_force", "evidence_ids": ids}, None, ev).reason, "single_model")

    def test_rfc2822_and_iso_dates(self):
        self.assertEqual(cf.parse_date("Sun, 22 Nov 2026 08:00:00 GMT"), date(2026, 11, 22))
        self.assertEqual(cf.parse_date("2026-11-22T08:00:00Z"), date(2026, 11, 22))


class Carry(unittest.TestCase):
    PREV = [{"id": "ceasefire", "label": "Ceasefire in force", "state": "yes", "violations": True, "carried": False,
             "last_confirmed": "2026-11-20T00:17:00Z", "checked_at": "2026-11-20T00:17:00Z", "evidence": [{"outlet": "bbc.com"}], "note": ""}]

    def test_carry_forward_when_no_new_evidence(self):
        out, d = cf.build_indicator(self.PREV, {"status": "unclear"}, {"status": "unclear"}, [], NOW, TODAY)
        self.assertEqual(out[0]["state"], "yes")
        self.assertTrue(out[0]["carried"])
        self.assertIn("3 days ago", out[0]["note"])

    def test_carry_expires(self):
        prev = [dict(self.PREV[0], last_confirmed="2026-11-01T00:00:00Z")]
        out, _ = cf.build_indicator(prev, {"status": "unclear"}, {"status": "unclear"}, [], NOW, TODAY)
        self.assertEqual(out[0]["state"], "unknown")

    def test_no_carry_when_sources_conflict_or_models_disagree(self):
        out, _ = cf.build_indicator(self.PREV, {"status": "in_force"}, {"status": "not_in_force"}, [], NOW, TODAY)
        self.assertEqual(out[0]["state"], "unknown")

    def test_one_model_down_keeps_previous(self):
        out, _ = cf.build_indicator(self.PREV, None, {"status": "in_force"}, [], NOW, TODAY)
        self.assertEqual((out[0]["state"], out[0]["carried"]), ("yes", True))


class Prompt(unittest.TestCase):
    def test_prompt_lists_ids_and_sentences(self):
        ev = cf.extract_evidence(GAZA, {}, TODAY, tier)
        p = cf.prompt_block(ev)
        self.assertIn("E1 [", p)
        self.assertIn("never write a quote", p)
        self.assertIn("(none found", cf.prompt_block([]))


# Real-world phrasings. (sentence, expected polarity). None = must NOT count either way.
# None entries are deliberate: a missed sentence only lowers recall (grey dot), a wrong one could show a false green/red.
PHRASES = [
    ("Israeli strikes killed six people in Gaza on Sunday, officials said, in the latest violation of the October ceasefire.", "holds"),
    ("The ceasefire in Gaza has largely held since it took effect in October, though violations continue.", "holds"),
    ("Thailand and Cambodia signed a ceasefire last October, but tensions remain.", "holds"),
    ("India and Pakistan have maintained a ceasefire since May 2025.", "holds"),           # 'May' is a month, not the word 'may'
    ("The ceasefire is holding, but the situation remains fragile, the UN envoy said.", "holds"),
    ("The ceasefire that ended the war remains in place, officials said.", "holds"),
    ("There is no ceasefire in Ukraine and the war continues, the Kremlin spokesman said.", "collapse"),
    ("Moscow has repeatedly rejected calls for an unconditional ceasefire.", "collapse"),
    ("Hamas said Israel has ended the ceasefire by resuming strikes.", "collapse"),
    ("Sudan's army rejected a humanitarian truce proposed by the Quad.", "collapse"),
    ("Ceasefire talks have stalled, with both sides blaming each other.", "collapse"),
    ("Iran's foreign minister said there was no ceasefire agreement with the United States.", "collapse"),
    ("Hostilities resumed after the ceasefire expired at midnight.", "collapse"),
    ("The Gaza ceasefire between Israel and Hamas collapsed on Sunday.", "collapse"),
    ("Trump said Russia and Ukraine should agree to an immediate ceasefire.", None),
    ("Zelensky said Ukraine is ready for a ceasefire if Russia stops its attacks.", None),
    ("Analysts warn the ceasefire could collapse within weeks.", None),
    ("A two-day truce around the Kyiv visit ended on Sunday.", None),                       # short pause: not evidence either way
    ("Russia announced a 30-hour Easter ceasefire.", None),                                 # must never turn a dot green
    ("The ceasefire that ended the 12-day war remains in place, Iranian officials said.", None),
    ("Hamas said there were no ceasefire violations this week.", None),                     # 'no ceasefire' != no ceasefire
    ("The ceasefire that ended the war collapsed on Sunday.", None),                        # known miss, safe direction
    ("Observers may see the ceasefire as fragile.", None),
    ("The ceasefire has not collapsed and remains in place.", None),
]


class RealPhrases(unittest.TestCase):
    def test_table(self):
        wrong = [(s, want, cf.polarity(s)) for s, want in PHRASES if cf.polarity(s) != want]
        self.assertEqual(wrong, [])


if __name__ == "__main__":
    unittest.main()
