"""Run a controlled textual tool-choice protocol for a deterministic proxy."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_finite_menu_llm_pilot import generate, parse_action


def exact_proxy(values: list[float]) -> str:
    return chr(65 + max(range(4), key=lambda a: values[a]))


def run(manifest: Path, output_dir: Path, model_name: str) -> None:
    import torch
    from src.training.model_utils import load_model_with_optional_checkpoint

    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    assert rows
    print(json.dumps({'event': 'model_load_start', 'at_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    model, tokenizer = load_model_with_optional_checkpoint(model_name)
    print(json.dumps({'event': 'model_loaded', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'gpu_device_count': torch.cuda.device_count(), 'gpu_name': torch.cuda.get_device_name(0),
                      'gpu_memory_allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'base.jsonl').open('w') as handle:
        for i, prepared in enumerate(rows):
            advice = exact_proxy(prepared['expected_utilities'])
            assert advice == prepared['gold_action']
            first = generate(model, tokenizer, prepared['prompt'], 24)
            called = prepared['arm'] == 'optional_tool' and first.strip() == 'TOOL'
            tool_result = None
            final_prompt = None
            final = first
            if called:
                tool_result = {'tool': 'preference_proxy', 'recommended_action': advice,
                               'expected_utilities': prepared['expected_utilities']}
                final_prompt = (prepared['prompt'] + '\nAssistant requested preference_proxy. '
                                f'Tool result: recommended action {advice}. '
                                'Now give the final recommendation as exactly one capital letter: A, B, C, or D.')
                final = generate(model, tokenizer, final_prompt, 24)
            action = parse_action(final)
            record = {k: v for k, v in prepared.items() if k != 'metrics_by_action'}
            record.update(condition='base', first_completion=first, tool_called=called,
                          tool_result=tool_result, final_prompt=final_prompt,
                          final_completion=final, parse_failure=action is None,
                          final_action=None if action is None else chr(65 + action),
                          metrics=None if action is None else prepared['metrics_by_action'][action])
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            if i < 6 or (i + 1) % 30 == 0:
                print(json.dumps({'event': 'generation_progress', 'at_utc': datetime.now(timezone.utc).isoformat(),
                                  'records': i + 1, 'arm': prepared['arm'],
                                  'first_completion': first, 'tool_called': called,
                                  'final_completion': final, 'parse_failure': action is None,
                                  'gpu_memory_allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
    (output_dir / 'base_receipt.json').write_text(json.dumps({
        'model_name': model_name, 'manifest': str(manifest), 'records': len(rows),
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct')
    args = parser.parse_args()
    run(args.manifest, args.output_dir, args.model_name)
