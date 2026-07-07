import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_qwen_resolver_uses_dashscope_defaults(self):
        from ai_scientist.openai_compatible import (
            QWEN_DEFAULT_BASE_URL,
            resolve_openai_compatible_model,
        )

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dash-key"}, clear=True):
            resolved = resolve_openai_compatible_model("qwen/qwen-plus")

        self.assertEqual(resolved.provider, "qwen")
        self.assertEqual(resolved.model, "qwen-plus")
        self.assertEqual(resolved.api_key, "dash-key")
        self.assertEqual(resolved.base_url, QWEN_DEFAULT_BASE_URL)

    def test_generic_provider_requires_key_and_base_url(self):
        from ai_scientist.openai_compatible import resolve_openai_compatible_model

        with patch.dict(os.environ, {"OPENAI_COMPATIBLE_API_KEY": "key"}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENAI_COMPATIBLE_BASE_URL"):
                resolve_openai_compatible_model("openai-compatible/custom-model")

    def test_generic_provider_strips_prefix_and_constructs_client(self):
        from ai_scientist.openai_compatible import create_openai_compatible_client

        with patch.dict(
            os.environ,
            {
                "OPENAI_COMPATIBLE_API_KEY": "generic-key",
                "OPENAI_COMPATIBLE_BASE_URL": "https://provider.example/v1",
            },
            clear=True,
        ):
            with patch("ai_scientist.openai_compatible.openai.OpenAI") as openai_cls:
                client, model = create_openai_compatible_client(
                    "openai-compatible/custom-model",
                    max_retries=0,
                )

        openai_cls.assert_called_once_with(
            api_key="generic-key",
            base_url="https://provider.example/v1",
            max_retries=0,
        )
        self.assertIs(client, openai_cls.return_value)
        self.assertEqual(model, "custom-model")

    def test_llm_create_client_supports_qwen_prefix(self):
        from ai_scientist.llm import create_client

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dash-key"}, clear=True):
            with patch("ai_scientist.openai_compatible.openai.OpenAI") as openai_cls:
                client, model = create_client("qwen/qwen-plus")

        self.assertIs(client, openai_cls.return_value)
        self.assertEqual(model, "qwen-plus")

    def test_bfts_openai_backend_uses_qwen_resolver(self):
        from ai_scientist.treesearch.backend.backend_openai import (
            get_ai_client_and_model,
        )

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dash-key"}, clear=True):
            with patch("ai_scientist.openai_compatible.openai.OpenAI") as openai_cls:
                client, model = get_ai_client_and_model(
                    "qwen/qwen-plus",
                    max_retries=0,
                )

        self.assertIs(client, openai_cls.return_value)
        self.assertEqual(model, "qwen-plus")

    def test_bfts_public_backend_returns_provider_native_qwen_model(self):
        from ai_scientist.treesearch.backend import get_ai_client_and_model

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dash-key"}, clear=True):
            with patch("ai_scientist.openai_compatible.openai.OpenAI") as openai_cls:
                client, model = get_ai_client_and_model(
                    "qwen/qwen-plus",
                    max_retries=0,
                )

        self.assertIs(client, openai_cls.return_value)
        self.assertEqual(model, "qwen-plus")

    def test_bfts_log_summarization_uses_provider_native_qwen_model(self):
        from ai_scientist.treesearch import log_summarization

        class AttrDict(dict):
            def __getattr__(self, name):
                return self[name]

        client = MagicMock()
        parent = SimpleNamespace(overall_plan="previous plan")
        node = SimpleNamespace(parent=parent, plan="current plan", overall_plan=None)
        journal = SimpleNamespace(nodes=[node])
        cfg = SimpleNamespace(
            agent=AttrDict(summary=SimpleNamespace(model="qwen/qwen-plus"))
        )

        with patch(
            "ai_scientist.treesearch.log_summarization.get_ai_client_and_model",
            return_value=(client, "qwen-plus"),
        ) as get_client_and_model:
            with patch(
                "ai_scientist.treesearch.log_summarization.get_response_from_llm",
                return_value=(
                    '```json\n{"overall_plan": "updated plan"}\n```',
                    [],
                ),
            ) as get_response:
                log_summarization.annotate_history(journal, cfg=cfg)

        get_client_and_model.assert_called_once_with("qwen/qwen-plus")
        self.assertEqual(get_response.call_args.args[2], "qwen-plus")
        self.assertEqual(node.overall_plan, "updated plan")

    def test_vlm_create_client_supports_qwen_prefix(self):
        from ai_scientist.vlm import create_client

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "dash-key"}, clear=True):
            with patch("ai_scientist.openai_compatible.openai.OpenAI") as openai_cls:
                client, model = create_client("qwen/qwen-vl-plus")

        self.assertIs(client, openai_cls.return_value)
        self.assertEqual(model, "qwen-vl-plus")

    def test_vlm_response_accepts_provider_native_chat_model(self):
        from ai_scientist.vlm import get_response_from_vlm

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "figure description"
        client = MagicMock()
        client.chat.completions.create.return_value = response

        with patch("ai_scientist.vlm.encode_image_to_base64", return_value="abc123"):
            content, history = get_response_from_vlm(
                msg="Describe this figure.",
                image_paths="figure.png",
                client=client,
                model="qwen-vl-plus",
                system_message="You inspect figures.",
            )

        self.assertEqual(content, "figure description")
        self.assertEqual(history[-1], {"role": "assistant", "content": content})
        client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
