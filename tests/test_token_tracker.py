import unittest
from types import SimpleNamespace


class TokenTrackerTests(unittest.TestCase):
    def setUp(self):
        from ai_scientist.utils.token_tracker import token_tracker

        token_tracker.reset()

    def tearDown(self):
        from ai_scientist.utils.token_tracker import token_tracker

        token_tracker.reset()

    def test_tracks_usage_when_prompt_token_details_are_missing(self):
        from ai_scientist.utils.token_tracker import token_tracker, track_token_usage

        def fake_llm_call(*, prompt, system_message):
            usage = SimpleNamespace(
                prompt_tokens=31,
                completion_tokens=17,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=10),
                prompt_tokens_details=None,
            )
            return SimpleNamespace(
                model="gpt-5.5",
                created=123,
                usage=usage,
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="OK")),
                ],
            )

        wrapped = track_token_usage(fake_llm_call)
        wrapped(
            prompt=[{"role": "user", "content": "Reply with OK only."}],
            system_message="You are a test harness.",
        )

        summary = token_tracker.get_summary()
        self.assertEqual(summary["gpt-5.5"]["tokens"]["prompt"], 31)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["completion"], 17)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["reasoning"], 10)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["cached"], 0)

    def test_tracks_usage_when_completion_token_details_are_missing(self):
        from ai_scientist.utils.token_tracker import token_tracker, track_token_usage

        def fake_llm_call(*, prompt, system_message):
            usage = SimpleNamespace(
                prompt_tokens=31,
                completion_tokens=17,
                completion_tokens_details=None,
                prompt_tokens_details=SimpleNamespace(cached_tokens=6),
            )
            return SimpleNamespace(
                model="gpt-5.5",
                created=123,
                usage=usage,
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="OK")),
                ],
            )

        wrapped = track_token_usage(fake_llm_call)
        wrapped(
            prompt=[{"role": "user", "content": "Reply with OK only."}],
            system_message="You are a test harness.",
        )

        summary = token_tracker.get_summary()
        self.assertEqual(summary["gpt-5.5"]["tokens"]["prompt"], 31)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["completion"], 17)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["reasoning"], 0)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["cached"], 6)

    def test_tracks_usage_when_optional_token_counts_are_none(self):
        from ai_scientist.utils.token_tracker import token_tracker, track_token_usage

        def fake_llm_call(*, prompt, system_message):
            usage = SimpleNamespace(
                prompt_tokens=31,
                completion_tokens=17,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=None),
                prompt_tokens_details=SimpleNamespace(cached_tokens=None),
            )
            return SimpleNamespace(
                model="gpt-5.5",
                created=123,
                usage=usage,
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="OK")),
                ],
            )

        wrapped = track_token_usage(fake_llm_call)
        wrapped(
            prompt=[{"role": "user", "content": "Reply with OK only."}],
            system_message="You are a test harness.",
        )

        summary = token_tracker.get_summary()
        self.assertEqual(summary["gpt-5.5"]["tokens"]["prompt"], 31)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["completion"], 17)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["reasoning"], 0)
        self.assertEqual(summary["gpt-5.5"]["tokens"]["cached"], 0)

    def test_rejects_non_chat_completion_response_with_base_url_hint(self):
        from ai_scientist.utils.token_tracker import track_token_usage

        def fake_llm_call(*, prompt, system_message):
            return "<!doctype html><title>Sub2API</title>"

        wrapped = track_token_usage(fake_llm_call)
        with self.assertRaisesRegex(ValueError, "OpenAI-compatible.*base_url.*/v1"):
            wrapped(
                prompt=[{"role": "user", "content": "Reply with OK only."}],
                system_message="You are a test harness.",
            )


if __name__ == "__main__":
    unittest.main()
