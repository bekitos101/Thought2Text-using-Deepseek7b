"""
test_model_utils.py
-------------------
Unit tests for model_utils.py.
No GPU, no downloaded models required — pure logic tests.

Run with:
    python -m pytest test_model_utils.py -v
or:
    python test_model_utils.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from model_utils import get_model_family, get_chat_messages, get_lora_target_modules


# ---------------------------------------------------------------------------
# get_model_family
# ---------------------------------------------------------------------------

class TestGetModelFamily:
    def test_deepseek_hf_path(self):
        assert get_model_family("deepseek-ai/deepseek-llm-7b-chat") == "deepseek"

    def test_deepseek_coder(self):
        assert get_model_family("deepseek-ai/deepseek-coder-7b-instruct-v1.5") == "deepseek"

    def test_deepseek_local_path(self):
        assert get_model_family("/models/deepseek-llm-7b-chat") == "deepseek"

    def test_mistral(self):
        assert get_model_family("mistralai/Mistral-7B-Instruct-v0.3") == "mistral"

    def test_llama(self):
        assert get_model_family("meta-llama/Meta-Llama-3-8B-Instruct") == "llama"

    def test_gemma(self):
        assert get_model_family("google/gemma-7b-it") == "gemma"

    def test_qwen(self):
        assert get_model_family("Qwen/Qwen2.5-7B-Instruct") == "qwen"

    def test_unknown_returns_default(self):
        assert get_model_family("some-unknown-org/some-model") == "default"

    def test_local_unknown_path(self):
        assert get_model_family("/home/user/my_custom_model") == "default"

    def test_case_insensitive(self):
        # Model names on HuggingFace can have mixed case
        assert get_model_family("MistralAI/Mistral-7B") == "mistral"
        assert get_model_family("DeepSeek-AI/DeepSeek-LLM-7B") == "deepseek"


# ---------------------------------------------------------------------------
# get_chat_messages
# ---------------------------------------------------------------------------

class TestGetChatMessages:
    def _roles(self, model_path):
        return [m["role"] for m in get_chat_messages(model_path)]

    def _contents(self, model_path):
        return [m["content"] for m in get_chat_messages(model_path)]

    # --- families WITH system role ---
    def test_deepseek_has_system_role(self):
        assert self._roles("deepseek-ai/deepseek-llm-7b-chat") == ["system", "user"]

    def test_mistral_has_system_role(self):
        assert self._roles("mistralai/Mistral-7B-Instruct-v0.3") == ["system", "user"]

    def test_llama_has_system_role(self):
        assert self._roles("meta-llama/Meta-Llama-3-8B-Instruct") == ["system", "user"]

    def test_qwen_has_system_role(self):
        assert self._roles("Qwen/Qwen2.5-7B-Instruct") == ["system", "user"]

    def test_default_has_system_role(self):
        assert self._roles("some-random/model") == ["system", "user"]

    # --- families WITHOUT system role ---
    def test_gemma_no_system_role(self):
        assert self._roles("google/gemma-7b-it") == ["user"]

    # --- content checks ---
    def test_user_content_contains_image_placeholder(self):
        for model in [
            "deepseek-ai/deepseek-llm-7b-chat",
            "google/gemma-7b-it",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]:
            msgs = get_chat_messages(model)
            user_msg = next(m for m in msgs if m["role"] == "user")
            assert "<image>" in user_msg["content"], f"<image> missing for {model}"
            assert "<label_string>" in user_msg["content"], f"<label_string> missing for {model}"

    def test_returns_new_list_each_call(self):
        # Mutations on the returned list must not affect subsequent calls
        msgs1 = get_chat_messages("mistralai/Mistral-7B-Instruct-v0.3")
        msgs1.append({"role": "assistant", "content": "hello"})
        msgs2 = get_chat_messages("mistralai/Mistral-7B-Instruct-v0.3")
        assert len(msgs2) == 2  # system + user only, no leaked assistant turn


# ---------------------------------------------------------------------------
# get_lora_target_modules
# ---------------------------------------------------------------------------

class TestGetLoraTargetModules:
    def test_deepseek_includes_gate_proj(self):
        modules = get_lora_target_modules("deepseek-ai/deepseek-llm-7b-chat")
        assert "gate_proj" in modules

    def test_deepseek_standard_attention_projections(self):
        modules = get_lora_target_modules("deepseek-ai/deepseek-llm-7b-chat")
        for m in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            assert m in modules

    def test_mistral_same_as_deepseek(self):
        assert (
            get_lora_target_modules("mistralai/Mistral-7B-Instruct-v0.3")
            == get_lora_target_modules("deepseek-ai/deepseek-llm-7b-chat")
        )

    def test_gemma_no_gate_proj(self):
        modules = get_lora_target_modules("google/gemma-7b-it")
        assert "gate_proj" not in modules

    def test_returns_new_list_each_call(self):
        # Mutations must not affect subsequent calls
        mods = get_lora_target_modules("mistralai/Mistral-7B-Instruct-v0.3")
        mods.clear()
        mods2 = get_lora_target_modules("mistralai/Mistral-7B-Instruct-v0.3")
        assert len(mods2) > 0

    def test_unknown_model_gets_default(self):
        mods = get_lora_target_modules("some-unknown/model")
        assert len(mods) > 0
        assert "q_proj" in mods


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [TestGetModelFamily, TestGetChatMessages, TestGetLoraTargetModules]
    passed = failed = 0

    for suite_cls in suites:
        suite = suite_cls()
        for name in [n for n in dir(suite_cls) if n.startswith("test_")]:
            try:
                getattr(suite, name)()
                print(f"  PASS  {suite_cls.__name__}.{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {suite_cls.__name__}.{name}")
                traceback.print_exc()
                failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
