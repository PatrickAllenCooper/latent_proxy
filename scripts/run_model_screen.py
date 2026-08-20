"""Cheap screening pass across every Foundry chat-capable deployment.

For a paired set of synthetic users (same seed, same env, across every
model), runs the LLM-driven elicitation loop ("base" condition) and
records alignment vs. the true optimal action, quality-floor violation
rate, and query/recommendation parse-failure rate. Also runs the
analytical (particle filter + EIG) condition once as a CPU-only reference
row -- no text generation involved, same seed/users, so it is directly
comparable.

I/O-bound (network calls), so models run concurrently via a thread pool
without contending with any CPU-bound experiment already running.

Usage:
    python scripts/run_model_screen.py --n-users 10 --max-rounds 5 \
        --env game --workers 4 --output-dir outputs/model_screen
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.agents.llm_elicitation import LLMElicitationConfig
from src.evaluation.dpo_study import DPOStudyConfig, run_dpo_study

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("model_screen")
logger.setLevel(logging.INFO)

# (label, backend, deployment) -- backend is "foundry-claude" (native
# Anthropic Messages API) or "foundry-openai" (unified Responses API,
# covers OpenAI-family + every third-party MaaS deployment on Foundry).
MODELS: list[tuple[str, str, str]] = [
    # Anthropic, native
    ("claude-haiku-4-5", "foundry-claude", "claude-haiku-4-5"),
    ("claude-sonnet-4-6", "foundry-claude", "claude-sonnet-4-6"),
    ("claude-opus-4-7", "foundry-claude", "claude-opus-4-7"),
    ("claude-opus-5", "foundry-claude", "claude-opus-5"),
    ("claude-fable-5", "foundry-claude", "claude-fable-5"),
    # OpenAI family
    ("gpt-4o-mini", "foundry-openai", "gpt-4o-mini"),
    ("gpt-4o", "foundry-openai", "gpt-4o"),
    ("gpt-5.4-nano", "foundry-openai", "gpt-5.4-nano"),
    ("gpt-5.4-mini", "foundry-openai", "gpt-5.4-mini"),
    ("gpt-5.2-chat", "foundry-openai", "gpt-5.2-chat"),
    ("gpt-5.6-sol", "foundry-openai", "gpt-5.6-sol"),  # unidentified deployment, screened opportunistically
    # Reasoning ablation pair (same family)
    ("DeepSeek-V3.2", "foundry-openai", "DeepSeek-V3.2"),
    ("DeepSeek-R1", "foundry-openai", "DeepSeek-R1"),
    # Cross-family / open-weight breadth
    ("DeepSeek-V4-Flash", "foundry-openai", "DeepSeek-V4-Flash"),
    ("Kimi-K2.5", "foundry-openai", "Kimi-K2.5"),
    ("Llama-3.3-70B-Instruct", "foundry-openai", "Llama-3.3-70B-Instruct"),
    ("Phi-4-multimodal-instruct", "foundry-openai", "Phi-4-multimodal-instruct"),
    ("grok-4-1-fast-reasoning", "foundry-openai", "grok-4-1-fast-reasoning"),
]


def _condition_result_to_dict(cr: Any) -> dict[str, Any]:
    d = asdict(cr)
    return d


def run_one_model(
    label: str, backend: str, deployment: str,
    *, env: str, n_users: int, max_rounds: int, seed: int,
    max_new_tokens: int, output_dir: Path,
) -> dict[str, Any]:
    log_path = output_dir / "completions" / f"{label}.jsonl"
    t0 = time.time()
    try:
        cfg = DPOStudyConfig(
            n_users=n_users,
            max_rounds=max_rounds,
            environments=[env],
            llm_config=LLMElicitationConfig(max_rounds=max_rounds, max_new_tokens=max_new_tokens),
            seed=seed,
            backend=backend,
            deployment=deployment,
            generation_log=str(log_path),
            conditions=["base"],
        )
        result = run_dpo_study(cfg)
        cr = result.per_env[env]["base"]
        elapsed = time.time() - t0
        record = {
            "label": label, "backend": backend, "deployment": deployment,
            "status": "ok", "elapsed_seconds": elapsed,
            **_condition_result_to_dict(cr),
        }
        logger.info(
            "%-28s align=%.3f viol=%.2f qfail=%.2f rfail=%.2f (%.1fs)",
            label, cr.mean_alignment, cr.mean_violation,
            cr.mean_query_parse_failure, cr.mean_rec_parse_failure, elapsed,
        )
        return record
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        logger.error("%-28s FAILED after %.1fs: %s: %s", label, elapsed, type(exc).__name__, exc)
        return {
            "label": label, "backend": backend, "deployment": deployment,
            "status": "error", "elapsed_seconds": elapsed,
            "error_type": type(exc).__name__, "error_message": str(exc),
        }


def run_analytical_reference(
    *, env: str, n_users: int, max_rounds: int, seed: int,
) -> dict[str, Any]:
    cfg = DPOStudyConfig(
        n_users=n_users, max_rounds=max_rounds, environments=[env],
        seed=seed, conditions=["analytical"],
    )
    result = run_dpo_study(cfg)
    cr = result.per_env[env]["analytical"]
    return {"label": "analytical (reference)", "status": "ok", **_condition_result_to_dict(cr)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="game", choices=["game", "stock", "supply_chain"])
    parser.add_argument("--n-users", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="outputs/model_screen")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma list of labels to screen (default: all MODELS)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    (out_dir / "completions").mkdir(parents=True, exist_ok=True)

    models = MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in MODELS if m[0] in wanted]

    logger.info("Running analytical reference (CPU only, no API calls)...")
    ref = run_analytical_reference(
        env=args.env, n_users=args.n_users, max_rounds=args.max_rounds, seed=args.seed,
    )
    logger.info("analytical reference: align=%.3f viol=%.2f", ref["mean_alignment"], ref["mean_violation"])

    logger.info("Screening %d models on %d workers...", len(models), args.workers)
    results: list[dict[str, Any]] = [ref]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one_model, label, backend, deployment,
                env=args.env, n_users=args.n_users, max_rounds=args.max_rounds,
                seed=args.seed, max_new_tokens=args.max_new_tokens, output_dir=out_dir,
            ): label
            for label, backend, deployment in models
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: (-1 if r["label"].startswith("analytical") else 0, -r.get("mean_alignment", -1)))

    summary = {
        "config": {
            "env": args.env, "n_users": args.n_users, "max_rounds": args.max_rounds,
            "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "n_models": len(models),
        },
        "results": results,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Model screen leaderboard ===")
    print(f"{'label':28s} {'align':>7s} {'viol':>6s} {'qfail':>6s} {'rfail':>6s} {'status':>8s}")
    for r in results:
        if r["status"] != "ok":
            print(f"{r['label']:28s} {'--':>7s} {'--':>6s} {'--':>6s} {'--':>6s} {'ERROR':>8s}  {r.get('error_message','')[:60]}")
            continue
        print(
            f"{r['label']:28s} {r['mean_alignment']:7.3f} {r['mean_violation']:6.2f} "
            f"{r['mean_query_parse_failure']:6.2f} {r['mean_rec_parse_failure']:6.2f} {'OK':>8s}"
        )
    logger.info("Wrote %s", out_dir / "summary.json")


if __name__ == "__main__":
    main()
