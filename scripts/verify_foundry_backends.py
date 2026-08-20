"""One-off live smoke test for the two Foundry text-generation backends.

Fires a single tiny request at each of claude-haiku-4-5 (Anthropic route)
and gpt-4o-mini (Responses API route) to confirm endpoint/auth/response
parsing actually work against the real resource. Never prints credential
values -- only completion text, status, and token usage.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents.text_backends import FoundryClaudeGenerator, FoundryOpenAIGenerator

PROMPT = "Reply with exactly one word: the capital of France."
LOG_PATH = REPO_ROOT / "outputs" / "foundry_verify.jsonl"


def try_one(label: str, build):
    print(f"--- {label} ---")
    try:
        gen = build()
        out = gen.generate(PROMPT)
        print(f"OK: {out!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    results = {}
    results["claude-haiku-4-5 (foundry-claude)"] = try_one(
        "claude-haiku-4-5 via FoundryClaudeGenerator",
        lambda: FoundryClaudeGenerator(
            "claude-haiku-4-5", max_tokens=20, log_path=LOG_PATH,
        ),
    )
    results["gpt-4o-mini (foundry-openai)"] = try_one(
        "gpt-4o-mini via FoundryOpenAIGenerator",
        lambda: FoundryOpenAIGenerator(
            "gpt-4o-mini", max_output_tokens=20, log_path=LOG_PATH,
        ),
    )
    results["Llama-3.3-70B-Instruct (foundry-openai, unified-gateway check)"] = try_one(
        "Llama-3.3-70B-Instruct via FoundryOpenAIGenerator (same endpoint as gpt-4o-mini)",
        lambda: FoundryOpenAIGenerator(
            "Llama-3.3-70B-Instruct", max_output_tokens=20, log_path=LOG_PATH,
        ),
    )

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"{k}: {'OK' if v else 'FAILED'}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
