"""
Thin OpenRouter wrapper.

One function that sends a chat completion request and returns the text.
`resolve_model_for_tier()` maps route.py's tier names to real model ids.
"""
import os
from typing import Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODEL = "openai/gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a legal document assistant. Answer the user's question "
    "using ONLY the contract text supplied in the context below. "
    "If the context does not contain enough information to answer, say "
    "so explicitly instead of guessing. Do not invent clauses, parties, "
    "dates, or figures that are not present in the supplied text. "
    "Where helpful, briefly point to the specific clause or section your "
    "answer is based on."
)

CASUAL_SYSTEM_PROMPT = (
    "You are a friendly assistant for a legal document tool. The user "
    "is making casual conversation, not asking about a specific "
    "document. Respond naturally and briefly."
)

# Tier -> real OpenRouter model id. Override any via .env.
_TIER_ENV_VARS = {
    "light": "MODEL_TIER_LIGHT",
    "standard": "MODEL_TIER_STANDARD",
    "strong": "MODEL_TIER_STRONG",
}

_TIER_DEFAULTS = {
    "light": "meta-llama/llama-3.2-3b-instruct",
    "standard": "meta-llama/llama-3.1-8b-instruct",
    "strong": "meta-llama/llama-3.1-70b-instruct",
}


def resolve_model_for_tier(tier: str) -> str:
    """Map a Level 2 tier name to a real OpenRouter model id.

    Reads the id from the matching env var (MODEL_TIER_LIGHT/STANDARD/
    STRONG) if it's set, otherwise falls back to a same-model-family
    default. An unrecognized tier name falls back to the standard tier
    rather than raising.
    """
    env_var = _TIER_ENV_VARS.get(tier, _TIER_ENV_VARS["standard"])
    default = _TIER_DEFAULTS.get(tier, _TIER_DEFAULTS["standard"])
    return os.environ.get(env_var) or default


def call(
    question: str,
    context: str,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Send one grounded question+context pair to OpenRouter and return
    the model's text response.

    Raises RuntimeError if OPENROUTER_API_KEY is not set, so a missing
    key fails loudly and immediately rather than silently returning a
    bad answer.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env "
            "and add your key."
        )

    resolved_model = model or os.environ.get("OPENROUTER_MODEL", FALLBACK_MODEL)
    resolved_system_prompt = system_prompt or SYSTEM_PROMPT
    user_content = (
        f"Contract context:\n{context}\n\nQuestion: {question}" if context else question
    )

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": resolved_system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
