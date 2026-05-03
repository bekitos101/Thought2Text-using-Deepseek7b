"""
model_utils.py
--------------
Centralised helpers for model-family detection, chat-template construction,
and LoRA target-module selection.

Every model-specific branch lives here so that datautils.py, inference.py,
and model.py stay clean and do not grow their own ad-hoc if/elif chains.

Supported families (detected from model name or path):
  - "deepseek"  : deepseek-ai/deepseek-llm-*-chat, deepseek-ai/deepseek-*
  - "gemma"     : google/gemma-*
  - "llama"     : meta-llama/*, llama-*
  - "mistral"   : mistralai/Mistral-*
  - "qwen"      : Qwen/*
  - "default"   : anything else (treated like llama/mistral)
"""

from __future__ import annotations

_FAMILY_KEYWORDS: list[tuple[str, str]] = [
    ("deepseek", "deepseek"),
    ("gemma",    "gemma"),
    ("llama",    "llama"),
    ("mistral",  "mistral"),
    ("qwen",     "qwen"),
]


def get_model_family(model_name_or_path: str) -> str:
    """
    Return a short family string from a HuggingFace model name or local path.

    Examples
    --------
    >>> get_model_family("deepseek-ai/deepseek-llm-7b-chat")
    'deepseek'
    >>> get_model_family("mistralai/Mistral-7B-Instruct-v0.3")
    'mistral'
    >>> get_model_family("google/gemma-7b-it")
    'gemma'
    >>> get_model_family("meta-llama/Meta-Llama-3-8B-Instruct")
    'llama'
    >>> get_model_family("Qwen/Qwen2.5-7B-Instruct")
    'qwen'
    >>> get_model_family("/local/some_unknown_model")
    'default'
    """
    lower = model_name_or_path.lower()
    for keyword, family in _FAMILY_KEYWORDS:
        if keyword in lower:
            return family
    # Fallback: read _name_or_path from config.json (handles generic local dirs)
    import os, json
    cfg = os.path.join(model_name_or_path, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg) as f:
                saved_name = json.load(f).get("_name_or_path", "").lower()
            for keyword, family in _FAMILY_KEYWORDS:
                if keyword in saved_name:
                    return family
        except (OSError, json.JSONDecodeError):
            pass
    return "default"


# ---------------------------------------------------------------------------
# Chat-message templates
# ---------------------------------------------------------------------------

# The <image> and <label_string> placeholders are preserved here — they are
# replaced at call sites (datautils.py / inference.py) after apply_chat_template
# splits the text around <image>.
_SYSTEM_PROMPT = "You are a helpful assistant."
_USER_CONTENT   = "<image> <label_string> Describe this image in one sentence:"

# Families that do NOT support a "system" role in their chat template.
_NO_SYSTEM_ROLE: set[str] = {"gemma"}


def get_chat_messages(model_name_or_path: str) -> list[dict]:
    """
    Return the messages list to pass to tokenizer.apply_chat_template().

    The list contains the *instruction* turns only (system + user).
    The caller appends the assistant turn when needed (fine-tuning),
    or uses add_generation_prompt=True for inference.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace repo id or local directory path.

    Returns
    -------
    list[dict]
        E.g. [{"role": "system", ...}, {"role": "user", ...}]
        or   [{"role": "user", ...}]  for models without a system role.

    Examples
    --------
    >>> msgs = get_chat_messages("google/gemma-7b-it")
    >>> [m["role"] for m in msgs]
    ['user']

    >>> msgs = get_chat_messages("deepseek-ai/deepseek-llm-7b-chat")
    >>> [m["role"] for m in msgs]
    ['system', 'user']

    >>> msgs = get_chat_messages("mistralai/Mistral-7B-Instruct-v0.3")
    >>> [m["role"] for m in msgs]
    ['system', 'user']
    """
    family = get_model_family(model_name_or_path)

    if family in _NO_SYSTEM_ROLE:
        return [
            {"role": "user", "content": _USER_CONTENT},
        ]

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _USER_CONTENT},
    ]


# ---------------------------------------------------------------------------
# LoRA target modules
# ---------------------------------------------------------------------------

# These are the attention / MLP projection layer names targeted by LoRA.
# DeepSeek-LLM-7B and DeepSeek-Coder use the same LLaMA-style naming.
_LORA_TARGETS: dict[str, list[str]] = {
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "llama":    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "mistral":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    "qwen":     ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    # Gemma uses the same projection names but no gate_proj in some variants.
    "gemma":    ["q_proj", "k_proj", "v_proj", "o_proj"],
    "default":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
}


def get_lora_target_modules(model_name_or_path: str) -> list[str]:
    """
    Return the list of module names to apply LoRA adapters to.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace repo id or local directory path.

    Returns
    -------
    list[str]

    Examples
    --------
    >>> get_lora_target_modules("deepseek-ai/deepseek-llm-7b-chat")
    ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj']
    >>> get_lora_target_modules("google/gemma-7b-it")
    ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    """
    family = get_model_family(model_name_or_path)
    return list(_LORA_TARGETS.get(family, _LORA_TARGETS["default"]))
