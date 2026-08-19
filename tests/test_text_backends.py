"""Tests for pluggable text-generation backends (stub, azure, local, selection)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agents.llm_elicitation import LLMElicitationConfig, LLMElicitationLoop
from src.agents.text_backends import (
    AzureChatGenerator,
    LocalHFGenerator,
    StubGenerator,
    TextGenerator,
    create_generator,
)
from src.environments.resource_game import ResourceStrategyGame
from src.training.serialization import AllocationSerializer
from src.training.synthetic_users import SyntheticUser, UserType


# ---------------------------------------------------------------------------
# StubGenerator
# ---------------------------------------------------------------------------

QUERY_PROMPT = """You are an advisor.

Total value: $10,000.00

Propose two distinct allocation strategies.
Format your response EXACTLY as:
Option A:
  safe: __%
  growth: __%
  aggressive: __%
  volatile: __%
Option B:
  safe: __%
  growth: __%
  aggressive: __%
  volatile: __%
"""

RECOMMEND_PROMPT = """You are an advisor.

Format your response EXACTLY as:
Recommended allocation:
  US Equities: __%
  Intl Equities: __%
  Bonds: __%
"""


def test_stub_query_output_parses_with_two_option_parser():
    from src.agents.llm_elicitation import parse_two_options

    stub = StubGenerator(seed=7)
    text = stub.generate(QUERY_PROMPT)
    assert "Option A:" in text and "Option B:" in text

    a, b = parse_two_options(text, 4)
    assert np.isclose(a.sum(), 1.0, atol=0.01)
    assert np.isclose(b.sum(), 1.0, atol=0.01)
    # Should be a real parse, not the uniform fallback for both options.
    assert not (np.allclose(a, 0.25) and np.allclose(b, 0.25))


def test_stub_recommend_output_parses_with_allocation_serializer():
    stub = StubGenerator(seed=7)
    text = stub.generate(RECOMMEND_PROMPT)
    assert text.startswith("Recommended allocation:")

    ser = AllocationSerializer(["US Equities", "Intl Equities", "Bonds"])
    alloc = ser.parse(text)
    assert alloc.shape == (3,)
    assert np.isclose(alloc.sum(), 1.0, atol=0.01)


def test_stub_is_deterministic():
    s1 = StubGenerator(seed=3)
    s2 = StubGenerator(seed=3)
    for _ in range(4):
        assert s1.generate(QUERY_PROMPT) == s2.generate(QUERY_PROMPT)


def test_stub_satisfies_protocol():
    assert isinstance(StubGenerator(), TextGenerator)


def test_stub_loop_end_to_end_two_users_three_rounds():
    """Full elicitation loop, 2 users x 3 rounds, entirely offline."""
    cfg = LLMElicitationConfig(max_rounds=3)
    loop = LLMElicitationLoop(config=cfg, generator=StubGenerator(seed=11))

    for u in range(2):
        env = ResourceStrategyGame()
        K = env.config.n_channels
        ut = UserType(gamma=0.6 + 0.1 * u, alpha=1.0, lambda_=1.5)
        user = SyntheticUser(ut, seed=u)

        result = loop.run(env, user, seed=42 + u)

        assert result.n_rounds == 3
        assert result.recommendation.shape == (K,)
        assert np.isclose(result.recommendation.sum(), 1.0, atol=0.01)
        assert len(result.per_round_recommendations) == 3
        for rec in result.per_round_recommendations:
            assert np.isclose(rec.sum(), 1.0, atol=0.01)
        for h in result.history:
            assert h["choice"] in (0, 1)
            assert np.isclose(h["option_a"].sum(), 1.0, atol=0.01)
            assert np.isclose(h["option_b"].sum(), 1.0, atol=0.01)
            assert "Option A" in h["raw_query"]


# ---------------------------------------------------------------------------
# AzureChatGenerator (openai client mocked; no network)
# ---------------------------------------------------------------------------


def _fake_response(text="Option A:\n  safe: 60%\n", finish="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40),
    )


class _FakeCompletions:
    def __init__(self, outcomes):
        # outcomes: list of responses or exceptions to raise, consumed in order
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_client(outcomes):
    completions = _FakeCompletions(outcomes)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


def _rate_limit_error():
    import httpx2
    import openai

    resp = httpx2.Response(
        429, request=httpx2.Request("POST", "https://unit.test/chat"),
    )
    return openai.RateLimitError("simulated 429", response=resp, body=None)


def test_azure_request_shape_and_jsonl_logging(tmp_path):
    log_path = tmp_path / "completions.jsonl"
    client, completions = _fake_client([_fake_response("hello world")])

    gen = AzureChatGenerator(
        "my-deployment",
        endpoint="https://unit.test",
        api_key="not-a-real-key",
        log_path=log_path,
        client=client,
    )

    out = gen.generate("What is your allocation?")
    assert out == "hello world"

    # Request shape
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "my-deployment"
    assert call["messages"] == [
        {"role": "user", "content": "What is your allocation?"}
    ]
    assert call["temperature"] == pytest.approx(0.3)
    assert call["max_tokens"] == 256

    # JSONL logging
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["timestamp"] == 0
    assert rec["deployment"] == "my-deployment"
    assert rec["finish_reason"] == "stop"
    assert rec["completion"] == "hello world"
    assert len(rec["prompt_sha256"]) == 64
    assert rec["usage"] == {"prompt_tokens": 120, "completion_tokens": 40}

    # Counter is monotonically increasing
    completions.outcomes.append(_fake_response("second"))
    gen.generate("another prompt")
    lines = log_path.read_text().strip().splitlines()
    assert json.loads(lines[1])["timestamp"] == 1


def test_azure_retries_on_rate_limit_then_succeeds(tmp_path, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "src.agents.text_backends.time.sleep", lambda s: sleeps.append(s),
    )

    client, completions = _fake_client([
        _rate_limit_error(),
        _rate_limit_error(),
        _fake_response("finally"),
    ])
    gen = AzureChatGenerator(
        "dep",
        endpoint="https://unit.test",
        api_key="k",
        log_path=tmp_path / "log.jsonl",
        client=client,
    )

    assert gen.generate("p") == "finally"
    assert len(completions.calls) == 3
    # Exponential backoff: 1s then 2s
    assert sleeps == [1.0, 2.0]
    # Only the successful completion is logged
    lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


def test_azure_gives_up_after_max_retries(tmp_path, monkeypatch):
    import openai

    monkeypatch.setattr("src.agents.text_backends.time.sleep", lambda s: None)

    client, completions = _fake_client([_rate_limit_error() for _ in range(5)])
    gen = AzureChatGenerator(
        "dep",
        endpoint="https://unit.test",
        api_key="k",
        log_path=tmp_path / "log.jsonl",
        max_retries=5,
        client=client,
    )

    with pytest.raises(openai.RateLimitError):
        gen.generate("p")
    assert len(completions.calls) == 5
    assert not (tmp_path / "log.jsonl").exists()


def test_azure_client_construction_kwargs(monkeypatch, tmp_path):
    import openai

    captured = {}

    def fake_azure_openai(**kwargs):
        captured.update(kwargs)
        client, _ = _fake_client([_fake_response("ok")])
        return client

    monkeypatch.setattr(openai, "AzureOpenAI", fake_azure_openai)

    gen = AzureChatGenerator(
        "dep",
        endpoint="https://unit.test",
        api_key="secret",
        api_version="2024-10-21",
        log_path=tmp_path / "log.jsonl",
    )
    assert gen.generate("p") == "ok"
    assert captured["azure_endpoint"] == "https://unit.test"
    assert captured["api_key"] == "secret"
    assert captured["api_version"] == "2024-10-21"
    assert captured["max_retries"] == 0


def test_azure_env_var_fallbacks(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "env-dep")

    gen = AzureChatGenerator(log_path=None)
    assert gen.endpoint == "https://env.test"
    assert gen.api_key == "env-key"
    assert gen.api_version == "2025-01-01-preview"
    assert gen.deployment == "env-dep"


def test_azure_requires_deployment(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    with pytest.raises(ValueError, match="deployment"):
        AzureChatGenerator(log_path=None)


# ---------------------------------------------------------------------------
# LocalHFGenerator delegates to the historical transformers path
# ---------------------------------------------------------------------------


@patch("src.agents.llm_elicitation._generate_text", return_value="generated!")
def test_local_generator_delegates_to_existing_path(mock_gen):
    model, tokenizer = MagicMock(), MagicMock()
    cfg = LLMElicitationConfig(max_new_tokens=64)
    gen = LocalHFGenerator(model, tokenizer, config=cfg)

    assert gen.generate("a prompt") == "generated!"
    mock_gen.assert_called_once_with(model, tokenizer, "a prompt", cfg)


def test_local_generator_requires_model_and_tokenizer():
    with pytest.raises(ValueError):
        LocalHFGenerator(None, None)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_create_generator_stub():
    gen = create_generator("stub", seed=5)
    assert isinstance(gen, StubGenerator)
    assert gen.seed == 5


def test_create_generator_local():
    model, tokenizer = MagicMock(), MagicMock()
    gen = create_generator("local", model=model, tokenizer=tokenizer)
    assert isinstance(gen, LocalHFGenerator)


def test_create_generator_azure(tmp_path):
    gen = create_generator(
        "azure", deployment="dep", log_path=tmp_path / "log.jsonl",
        temperature=0.7, max_tokens=128,
    )
    assert isinstance(gen, AzureChatGenerator)
    assert gen.deployment == "dep"
    assert gen.temperature == pytest.approx(0.7)
    assert gen.max_tokens == 128


def test_create_generator_unknown():
    with pytest.raises(ValueError, match="Unknown text backend"):
        create_generator("carrier-pigeon")


# ---------------------------------------------------------------------------
# DPO study wiring
# ---------------------------------------------------------------------------


def test_resolve_conditions_defaults_and_validation():
    from src.evaluation.dpo_study import DPOStudyConfig, _resolve_conditions

    assert _resolve_conditions(DPOStudyConfig()) == ["analytical", "base"]
    assert _resolve_conditions(
        DPOStudyConfig(phase2_checkpoint="ckpt")
    ) == ["analytical", "base", "dpo_phase2"]
    assert _resolve_conditions(
        DPOStudyConfig(conditions=["analytical", "random"])
    ) == ["analytical", "random"]
    with pytest.raises(ValueError, match="Unknown conditions"):
        _resolve_conditions(DPOStudyConfig(conditions=["generic"]))


def test_dpo_study_rejects_dpo_conditions_on_stub_backend():
    from src.evaluation.dpo_study import DPOStudyConfig, run_dpo_study

    cfg = DPOStudyConfig(
        n_users=1, max_rounds=1, environments=["game"],
        backend="stub", conditions=["dpo_phase1"],
        phase1_checkpoint="ckpt",
    )
    with pytest.raises(ValueError, match="backend='local'"):
        run_dpo_study(cfg)


def test_dpo_study_base_condition_with_stub_backend():
    """base-LLM condition runs end-to-end through the stub backend."""
    from src.agents.llm_elicitation import LLMElicitationConfig
    from src.evaluation.dpo_study import DPOStudyConfig, run_dpo_study

    cfg = DPOStudyConfig(
        n_users=2, max_rounds=3, environments=["game"],
        backend="stub", conditions=["base"],
        llm_config=LLMElicitationConfig(max_rounds=3),
    )
    result = run_dpo_study(cfg)

    assert set(result.per_env.keys()) == {"game"}
    cr = result.per_env["game"]["base"]
    assert cr.condition == "base"
    assert len(cr.alignment_scores) == 2
    assert all(np.isfinite(s) for s in cr.alignment_scores)
    assert all(len(pr) == 3 for pr in cr.per_round_alignments)
    assert result.config["backend"] == "stub"
