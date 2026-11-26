"""Date parsing, Guard-2 date source, Cloudflare quota breaker, Gemini model chain, A/B independence.
No network, no keys. Run: python -m unittest discover -s tests -v"""
import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
for k in ("TAVILY_API_KEY", "GROQ_API_KEY", "CF_ACCOUNT_ID", "CF_API_TOKEN"):
    os.environ.setdefault(k, "test")
os.environ["CEASEFIRE_FETCH"] = "0"

import update  # noqa: E402


class DateParsing(unittest.TestCase):
    def test_all_formats_the_pipeline_can_see(self):
        want = datetime(2026, 11, 25, 18, 53, 38, tzinfo=timezone.utc)
        for v in ("Wed, 25 Nov 2026 18:53:38 GMT",          # Tavily (RFC-2822), the format that was silently dropped
                  "2026-11-25T18:53:38Z", "2026-11-25T18:53:38+00:00", "2026-11-25T20:53:38+02:00", "2026-11-25T18:53:38"):
            self.assertEqual(update.parse_pub_date(v), want, v)
        self.assertEqual(update.parse_pub_date("2026-11-25"), datetime(2026, 11, 25, tzinfo=timezone.utc))

    def test_garbage_never_raises(self):
        for v in (None, "", "   ", "garbage", 12345, "Wed, 99 Foo 2026", "2026-13-45"):
            self.assertIsNone(update.parse_pub_date(v), v)

    def test_latest_not_earliest_and_ignores_future_junk(self):
        now = datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc)
        res = [{"url": "https://a.com/1", "published_date": "Thu, 19 Nov 2026 06:00:00 GMT"},
               {"url": "https://b.com/2", "published_date": "Wed, 25 Nov 2026 15:03:21 GMT"},
               {"url": "https://c.com/3", "published_date": "Tue, 24 Nov 2026 10:43:00 GMT"},
               {"url": "https://d.com/4", "published_date": "Mon, 01 Jan 2029 00:00:00 GMT"},   # bad metadata
               {"url": "https://e.com/5"}]
        dt, meta = update.latest_source_time_from_results(res, now=now)
        self.assertEqual(dt, datetime(2026, 11, 25, 15, 3, 21, tzinfo=timezone.utc))
        self.assertEqual(meta["url"], "https://b.com/2")
        self.assertEqual(update.latest_source_time_from_results([], now=now), (None, None))
        self.assertEqual(update.latest_source_time_from_results([{"url": "x", "published_date": "nope"}], now=now), (None, None))

    def test_guard2_would_keep_a_recent_real_event(self):
        """Event on Nov 24, articles Nov 19..25: the newest article (Nov 25) proves the event can have been reported."""
        res = [{"url": f"https://x.com/{d}", "published_date": f"2026-11-{d}T10:00:00Z"} for d in (19, 21, 23, 24, 25)]
        newest, _ = update.latest_source_time_from_results(res, now=datetime(2026, 11, 26, tzinfo=timezone.utc))
        event = datetime(2026, 11, 24).date()
        self.assertFalse(newest.date() < event - timedelta(days=1))


class FakeResp:
    def __init__(self, status=200, text="", body=None):
        self.status_code, self.text, self._body = status, text, body
        self.ok = 200 <= status < 300
        self.headers = {}

    def json(self):
        return self._body


QUOTA = ('{"errors":[{"message":"AiError: AiError: you have used up your daily free allocation of 10,000 neurons, '
         'please upgrade to Cloudflare\'s Workers Paid plan if you would like to continue usage.","code":4006}],"success":false}')


class CloudflareBreaker(unittest.TestCase):
    def setUp(self):
        self.post = update.requests.post
        self.sleep = update.time.sleep
        update.time.sleep = lambda s: None
        update.CF_STATE.update({"dead": "", "calls": 0, "tokens_in": 0, "tokens_out": 0})
        self.calls = 0

    def tearDown(self):
        update.requests.post = self.post
        update.time.sleep = self.sleep
        update.CF_STATE.update({"dead": "", "calls": 0, "tokens_in": 0, "tokens_out": 0})

    def test_quota_error_stops_all_further_calls_in_the_run(self):
        def post(*a, **k):
            self.calls += 1
            return FakeResp(429, QUOTA)
        update.requests.post = post
        with self.assertRaises(RuntimeError):
            update.cloudflare("m", "p")
        self.assertEqual(self.calls, 1)                      # no pointless retries on a quota error
        self.assertIn("daily free allocation", update.CF_STATE["dead"])
        with self.assertRaises(RuntimeError) as cm:
            update.cloudflare("m", "p")
        self.assertIn("skipped", str(cm.exception))
        self.assertEqual(self.calls, 1)                      # and no HTTP request at all afterwards

    def test_transient_429_is_retried_then_succeeds_and_usage_is_counted(self):
        seq = [FakeResp(429, "rate limited"), FakeResp(503, "unavailable"),
               FakeResp(200, "", {"success": True, "result": {"response": '{"current": "x"}',
                                                              "usage": {"prompt_tokens": 6000, "completion_tokens": 800}}})]
        def post(*a, **k):
            self.calls += 1
            return seq.pop(0)
        update.requests.post = post
        self.assertEqual(update.cloudflare("m", "p"), {"current": "x"})
        self.assertEqual((self.calls, update.CF_STATE["dead"]), (3, ""))
        self.assertAlmostEqual(update.cloudflare_neurons_used(), 6000 * 26668 / 1e6 + 800 * 204805 / 1e6, places=3)

    def test_other_errors_fail_fast(self):
        update.requests.post = lambda *a, **k: (setattr(self, "calls", self.calls + 1), FakeResp(401, "bad token"))[1]
        with self.assertRaises(RuntimeError):
            update.cloudflare("m", "p")
        self.assertEqual((self.calls, update.CF_STATE["dead"]), (1, ""))


class GeminiChain(unittest.TestCase):
    def setUp(self):
        self.gem, self.key, self.models = update.gemini, update.GEMINI_API_KEY, update.GEMINI_MODELS
        update.GEMINI_API_KEY = "k"
        update.GEMINI_MODELS = ["gemini-new", "gemini-mid", "gemini-old"]
        update._GEMINI_DEAD.clear()
        self.seen = []

    def tearDown(self):
        update.gemini, update.GEMINI_API_KEY, update.GEMINI_MODELS = self.gem, self.key, self.models
        update._GEMINI_DEAD.clear()

    def test_unavailable_model_is_skipped_for_the_rest_of_the_run(self):
        def fake(model, prompt, max_tokens=900, json_mode=True):
            self.seen.append((model, max_tokens))
            if model == "gemini-new":
                raise RuntimeError("Gemini HTTP 404: model no longer available to new users")
            return {"current": model}
        update.gemini = fake
        self.assertEqual(update.gemini_chain("p", max_tokens=900), {"current": "gemini-mid"})
        self.assertEqual(update.gemini_chain("p", max_tokens=1200), {"current": "gemini-mid"})
        self.assertEqual([m for m, _ in self.seen], ["gemini-new", "gemini-mid", "gemini-mid"])   # 404 model tried only once
        self.assertTrue(all(t >= update.GEMINI_MIN_OUTPUT_TOKENS for _, t in self.seen))          # room for hidden thinking tokens

    def test_400_does_not_blacklist_a_model_but_moves_on(self):
        def fake(model, *a, **k):
            if model == "gemini-new":
                raise RuntimeError("Gemini HTTP 400: bad request")
            return {"current": model}
        update.gemini = fake
        self.assertEqual(update.gemini_chain("p"), {"current": "gemini-mid"})
        self.assertNotIn("gemini-new", update._GEMINI_DEAD)

    def test_all_fail_and_no_key(self):
        update.gemini = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Gemini HTTP 429: quota"))
        with self.assertRaises(RuntimeError):
            update.gemini_chain("p")
        with self.assertRaises(RuntimeError):
            update.gemini_chain("p")                          # all marked dead: fails fast, no calls
        update.GEMINI_API_KEY = ""
        with self.assertRaises(RuntimeError) as cm:
            update.gemini_chain("p")
        self.assertIn("GEMINI_API_KEY", str(cm.exception))


class Independence(unittest.TestCase):
    """A and B must never be the same model family; the source label must name who really answered."""
    def setUp(self):
        self.saved = (update.cloudflare, update.groq, update.gemini_chain)
        update.CF_STATE.update({"dead": "", "calls": 0, "tokens_in": 0, "tokens_out": 0})
        self.down = set()
        def cf(m, p, max_tokens=1200):
            if "cloudflare" in self.down:
                raise RuntimeError("Cloudflare HTTP 429: daily free allocation")
            return {"current": "llama"}
        def groq(m, p, max_tokens=900, json_mode=True, retries=2, total_timeout=None):
            key = "gpt-oss" if "gpt-oss" in m else "groq"
            if key in self.down:
                raise RuntimeError("Groq 429")
            return {"current": key}
        def gem(p, max_tokens=900, json_mode=True):
            if "gemini" in self.down:
                raise RuntimeError("Gemini down")
            return {"current": "gemini"}
        update.cloudflare, update.groq, update.gemini_chain = cf, groq, gem

    def tearDown(self):
        update.cloudflare, update.groq, update.gemini_chain = self.saved
        update.CF_STATE.update({"dead": "", "calls": 0, "tokens_in": 0, "tokens_out": 0})

    def pair(self):
        taken = set()
        out = []
        for role in ("A", "B"):
            try:
                out.append(update.run_researcher(role, "p", taken)[1])
            except RuntimeError:
                out.append(None)
        return tuple(out)

    def test_normal(self):
        self.assertEqual(self.pair(), ("cloudflare", "groq"))

    def test_cloudflare_out_gemini_replaces_A(self):
        self.down = {"cloudflare"}
        self.assertEqual(self.pair(), ("gemini", "groq"))

    def test_cloudflare_breaker_skips_A_without_calling(self):
        update.CF_STATE["dead"] = "quota"
        self.assertEqual(self.pair(), ("gemini", "groq"))

    def test_groq_out_gemini_replaces_B(self):
        self.down = {"groq"}
        self.assertEqual(self.pair(), ("cloudflare", "gemini"))

    def test_cloudflare_and_groq_out_never_two_geminis(self):
        self.down = {"cloudflare", "groq"}
        a, b = self.pair()
        self.assertEqual((a, b), ("gemini", "gpt-oss"))
        self.assertNotEqual(update.FAMILY[a], update.FAMILY[b])

    def test_everything_but_gemini_out_gives_single_model_not_a_fake_pair(self):
        self.down = {"cloudflare", "groq", "gpt-oss"}
        self.assertEqual(self.pair(), ("gemini", None))

    def test_non_object_answers_are_ignored(self):
        update.cloudflare = lambda *a, **k: ["not", "an", "object"]
        self.assertEqual(self.pair(), ("gemini", "groq"))


if __name__ == "__main__":
    unittest.main()
