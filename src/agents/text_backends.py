"""Text-generation backends for LLM-driven elicitation.

Provides a small abstraction over "prompt in, text out" so the elicitation
loop can run against:

- a local HuggingFace transformers model (the existing GPU path, unchanged),
- an Azure OpenAI chat deployment (API path, no local GPU needed),
- a Microsoft Foundry Claude deployment (native Anthropic Messages API),
- a deterministic stub (offline tests / dry runs).

All imports of heavy or optional dependencies (torch, transformers, openai,
anthropic) are deferred so this module imports cleanly in minimal
environments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_AZURE_API_VERSION = "2024-10-21"

_RETRYABLE_STATUS_HINT = "rate-limit or transient server error"


@runtime_checkable
class TextGenerator(Protocol):
    """Anything that maps a fully formatted prompt string to completion text."""

    def generate(self, prompt: str) -> str:
        ...


class LocalHFGenerator:
    """Wraps the existing local transformers generation path.

    Delegates to :func:`src.agents.llm_elicitation._generate_text`, so the
    behavior on CURC GPU nodes is byte-for-byte identical to the historical
    code path.
    """

    def __init__(self, model: Any, tokenizer: Any, config: Any | None = None) -> None:
        if model is None or tokenizer is None:
            raise ValueError("LocalHFGenerator requires a model and tokenizer")
        self.model = model
        self.tokenizer = tokenizer
        if config is None:
            from src.agents.llm_elicitation import LLMElicitationConfig

            config = LLMElicitationConfig()
        self.config = config

    def generate(self, prompt: str) -> str:
        # Late import + module-attribute lookup so test patches of
        # src.agents.llm_elicitation._generate_text keep working.
        from src.agents import llm_elicitation

        return llm_elicitation._generate_text(
            self.model, self.tokenizer, prompt, self.config,
        )


class AzureChatGenerator:
    """Azure OpenAI chat-completions backend.

    Config resolution order: constructor argument, then environment variable.

    - endpoint:    AZURE_OPENAI_ENDPOINT
    - api_key:     AZURE_OPENAI_API_KEY
    - api_version: AZURE_OPENAI_API_VERSION (default ``2024-10-21``)
    - deployment:  AZURE_OPENAI_DEPLOYMENT

    Every raw completion is appended to a JSONL log with a monotonically
    increasing ``timestamp`` counter (no wall-clock), the SHA-256 of the
    prompt, the deployment name, and the finish reason.
    """

    def __init__(
        self,
        deployment: str | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 256,
        log_path: str | Path | None = "outputs/azure_completions.jsonl",
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self.deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or DEFAULT_AZURE_API_VERSION
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.log_path = Path(log_path) if log_path else None
        self.max_retries = max(1, int(max_retries))
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._client = client
        self._counter = 0

        if not self.deployment:
            raise ValueError(
                "AzureChatGenerator needs a deployment name "
                "(constructor arg or AZURE_OPENAI_DEPLOYMENT)"
            )

    # -- client ------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.endpoint or not self.api_key:
                raise ValueError(
                    "AzureChatGenerator needs an endpoint and api key "
                    "(constructor args or AZURE_OPENAI_ENDPOINT / "
                    "AZURE_OPENAI_API_KEY)"
                )
            import openai

            # max_retries=0: retry policy is handled here, not in the client.
            self._client = openai.AzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
                max_retries=0,
            )
        return self._client

    # -- generation --------------------------------------------------------

    def generate(self, prompt: str) -> str:
        import openai

        retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,  # includes APITimeoutError
            openai.InternalServerError,
        )

        client = self._get_client()
        delay = self.backoff_base
        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                break
            except retryable as exc:
                if attempt >= self.max_retries:
                    logger.error(
                        "Azure generation failed after %d attempts: %s",
                        attempt, exc,
                    )
                    raise
                logger.warning(
                    "Azure %s (attempt %d/%d), backing off %.1fs: %s",
                    _RETRYABLE_STATUS_HINT, attempt, self.max_retries, delay, exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, self.backoff_max)

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        self._log_completion(
            prompt=prompt,
            completion=text,
            finish_reason=getattr(choice, "finish_reason", None),
            usage={
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            } if usage is not None else None,
        )
        return text

    def _log_completion(
        self,
        *,
        prompt: str,
        completion: str,
        finish_reason: str | None,
        usage: dict[str, Any] | None,
    ) -> None:
        if self.log_path is None:
            return
        record = {
            "timestamp": self._counter,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "deployment": self.deployment,
            "finish_reason": finish_reason,
            "completion": completion,
            "prompt_chars": len(prompt),
            "usage": usage,
        }
        self._counter += 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


class FoundryClaudeGenerator:
    """Microsoft Foundry, native Anthropic Messages API backend.

    For Claude deployments on Azure AI Foundry (``claude-opus-5``,
    ``claude-fable-5``, ``claude-opus-4-7``, ``claude-sonnet-4-6``,
    ``claude-haiku-4-5``, ...), reached via the ``anthropic`` SDK's
    ``AnthropicFoundry`` client rather than the OpenAI-compatible route
    ``AzureChatGenerator`` uses. Config resolution order: constructor
    argument, then environment variable.

    - endpoint:    FOUNDRY_ANTHROPIC_ENDPOINT
      (e.g. ``https://<resource>.services.ai.azure.com/anthropic``)
    - api_key:     FOUNDRY_ANTHROPIC_API_KEY
    - deployment:  FOUNDRY_ANTHROPIC_DEPLOYMENT (the model name, e.g. ``claude-opus-5``)

    Logs each raw completion the same way ``AzureChatGenerator`` does.
    """

    def __init__(
        self,
        deployment: str | None = None,
        *,
        endpoint: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 256,
        log_path: str | Path | None = "outputs/azure_completions.jsonl",
        max_retries: int = 5,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self.deployment = deployment or os.environ.get("FOUNDRY_ANTHROPIC_DEPLOYMENT")
        self.endpoint = endpoint or os.environ.get("FOUNDRY_ANTHROPIC_ENDPOINT")
        self.api_key = api_key or os.environ.get("FOUNDRY_ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.log_path = Path(log_path) if log_path else None
        self.max_retries = max(1, int(max_retries))
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._client = client
        self._counter = 0

        if not self.deployment:
            raise ValueError(
                "FoundryClaudeGenerator needs a deployment name "
                "(constructor arg or FOUNDRY_ANTHROPIC_DEPLOYMENT)"
            )

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.endpoint or not self.api_key:
                raise ValueError(
                    "FoundryClaudeGenerator needs an endpoint and api key "
                    "(constructor args or FOUNDRY_ANTHROPIC_ENDPOINT / "
                    "FOUNDRY_ANTHROPIC_API_KEY)"
                )
            from anthropic import AnthropicFoundry

            self._client = AnthropicFoundry(
                api_key=self.api_key, base_url=self.endpoint, max_retries=0,
            )
        return self._client

    def generate(self, prompt: str) -> str:
        import anthropic

        retryable = (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        )

        client = self._get_client()
        delay = self.backoff_base
        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.messages.create(
                    model=self.deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                )
                break
            except retryable as exc:
                if attempt >= self.max_retries:
                    logger.error(
                        "Foundry Claude generation failed after %d attempts: %s",
                        attempt, exc,
                    )
                    raise
                logger.warning(
                    "Foundry Claude %s (attempt %d/%d), backing off %.1fs: %s",
                    _RETRYABLE_STATUS_HINT, attempt, self.max_retries, delay, exc,
                )
                time.sleep(delay)
                delay = min(delay * 2, self.backoff_max)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(response, "usage", None)
        self._log_completion(
            prompt=prompt,
            completion=text,
            finish_reason=getattr(response, "stop_reason", None),
            usage={
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            } if usage is not None else None,
        )
        return text

    def _log_completion(
        self,
        *,
        prompt: str,
        completion: str,
        finish_reason: str | None,
        usage: dict[str, Any] | None,
    ) -> None:
        if self.log_path is None:
            return
        record = {
            "timestamp": self._counter,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "deployment": self.deployment,
            "finish_reason": finish_reason,
            "completion": completion,
            "prompt_chars": len(prompt),
            "usage": usage,
        }
        self._counter += 1
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


_CHANNEL_LINE_RE = re.compile(r"^\s*(.+?):\s*__%\s*$", re.MULTILINE)


class StubGenerator:
    """Deterministic canned responses for offline tests and dry runs.

    Reads the channel-name template lines (``name: __%``) out of the prompt,
    then emits allocations in exactly the format the existing parsers
    (``parse_two_options`` and ``AllocationSerializer.parse``) accept.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.n_calls = 0

    @staticmethod
    def _channel_names(prompt: str) -> list[str]:
        names: list[str] = []
        for m in _CHANNEL_LINE_RE.finditer(prompt):
            name = m.group(1).strip()
            if name and name not in names:
                names.append(name)
        return names or ["safe", "growth", "aggressive", "volatile"]

    def _allocation(self, n: int) -> np.ndarray:
        vals = self._rng.dirichlet(np.ones(n))
        return vals

    @staticmethod
    def _format_lines(names: list[str], alloc: np.ndarray) -> list[str]:
        return [f"  {name}: {alloc[i] * 100:.0f}%" for i, name in enumerate(names)]

    def generate(self, prompt: str) -> str:
        self.n_calls += 1
        names = self._channel_names(prompt)
        n = len(names)

        if "Recommended allocation" in prompt:
            lines = ["Recommended allocation:"]
            lines.extend(self._format_lines(names, self._allocation(n)))
            return "\n".join(lines)

        lines = ["Option A:"]
        lines.extend(self._format_lines(names, self._allocation(n)))
        lines.append("Option B:")
        lines.extend(self._format_lines(names, self._allocation(n)))
        return "\n".join(lines)


def create_generator(
    backend: str,
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    config: Any | None = None,
    deployment: str | None = None,
    log_path: str | Path | None = None,
    temperature: float = 0.3,
    max_tokens: int = 256,
    seed: int = 0,
) -> TextGenerator:
    """Build a TextGenerator for the named backend: local, azure, or stub."""
    backend = backend.lower()
    if backend == "local":
        return LocalHFGenerator(model, tokenizer, config=config)
    if backend == "azure":
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if log_path is not None:
            kwargs["log_path"] = log_path
        return AzureChatGenerator(deployment, **kwargs)
    if backend == "foundry-claude":
        kwargs = {"max_tokens": max_tokens}
        if log_path is not None:
            kwargs["log_path"] = log_path
        return FoundryClaudeGenerator(deployment, **kwargs)
    if backend == "stub":
        return StubGenerator(seed=seed)
    raise ValueError(
        f"Unknown text backend: {backend!r} "
        "(expected local/azure/foundry-claude/stub)"
    )
