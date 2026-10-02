"""Compare how an identical exact-proxy result is handed back to the model."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_finite_menu_llm_pilot import parse_action
from scripts.run_proxy_tool_choice import exact_proxy


def messages(row: dict, arm: str) -> list[dict]:
    advice = exact_proxy(row['expected_utilities'])
    prompt = row['prompt']
    assert row['gold_action'] == advice
    if arm == 'concatenated':
        return [{'role': 'user', 'content': prompt + '\nAssistant requested preference_proxy. '
                f'Tool result: recommended action {advice}. '
                'Now give the final recommendation as exactly one capital letter: A, B, C, or D.'}]
    if arm == 'compact_user':
        return [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': 'TOOL'},
                {'role': 'user', 'content': f'preference_proxy result: {advice}. '
                 'Final answer: copy that recommended action as one capital letter.'}]
    if arm == 'tool_role':
        return [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': 'TOOL'},
                {'role': 'tool', 'content': f'preference_proxy recommends {advice}'},
                {'role': 'user', 'content': 'Give the final recommendation as exactly one capital letter: A, B, C, or D.'}]
    raise ValueError(arm)


def run(source: Path, output: Path, model_name: str, limit: int | None) -> None:
    import torch
    from src.training.model_utils import load_model_with_optional_checkpoint

    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    rows = [r for r in rows if r['arm'] == 'optional_tool']
    assert len(rows) == 64
    if limit is not None:
        rows = rows[:limit]
    # Template compatibility and prompt inspection are CPU work, before the GPU job.
    print(json.dumps({'event': 'model_load_start', 'at_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    model, tokenizer = load_model_with_optional_checkpoint(model_name)
    print(json.dumps({'event': 'model_loaded', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'gpu_device_count': torch.cuda.device_count(), 'gpu_name': torch.cuda.get_device_name(0),
                      'gpu_memory_allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    count = 0
    with (output / 'base.jsonl').open('w') as handle:
        for row in rows:
            for arm in ('concatenated', 'compact_user', 'tool_role'):
                msg = messages(row, arm)
                encoded = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                                        return_tensors='pt').to(model.device)
                with torch.inference_mode():
                    generated = model.generate(encoded, max_new_tokens=24, do_sample=False,
                                               pad_token_id=tokenizer.eos_token_id)
                completion = tokenizer.decode(generated[0, encoded.shape[-1]:], skip_special_tokens=True)
                action = parse_action(completion)
                record = {'user_id': row['user_id'], 'scenario_id': row['scenario_id'],
                          'permutation_id': row['permutation_id'], 'arm': arm,
                          'gold_action': row['gold_action'], 'expected_utilities': row['expected_utilities'],
                          'source_first_completion': 'TOOL', 'messages': msg,
                          'rendered_prompt': tokenizer.decode(encoded[0], skip_special_tokens=False),
                          'completion': completion, 'final_action': None if action is None else chr(65 + action),
                          'parse_failure': action is None,
                          'metrics': None if action is None else row['metrics_by_action'][action]}
                handle.write(json.dumps(record) + '\n')
                handle.flush()
                count += 1
                if count <= 6 or count % 24 == 0:
                    print(json.dumps({'event': 'generation_progress', 'at_utc': datetime.now(timezone.utc).isoformat(),
                                      'records': count, 'arm': arm, 'completion': completion,
                                      'gpu_memory_allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
    (output / 'base_receipt.json').write_text(json.dumps({'model_name': model_name, 'source': str(source),
        'records': count, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.source, args.output_dir, args.model_name, args.limit)
