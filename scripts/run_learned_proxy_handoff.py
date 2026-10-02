"""Prepare learned-proxy advice on CPU and measure its LLM handoff."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def prepare(source: Path, checkpoint: Path, output: Path) -> None:
    import torch
    from src.evaluation.preference_benchmark import expected_utilities, make_scenario
    from src.training.finite_menu_ppo import FiniteMenuPolicy
    from src.training.synthetic_users import UserType

    policy = FiniteMenuPolicy()
    policy.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    policy.eval()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    rows = [r for r in rows if r['arm'] == 'optional_tool']
    assert len(rows) == 64
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for row in rows:
            theta = UserType(**row['true_theta'])
            scenario = make_scenario(int(row['scenario_id'].split('-')[1]))
            scores = expected_utilities(scenario, theta)
            shift = row['permutation_id']
            assert max(abs(float(scores[i]) - row['expected_utilities'][(i + shift) % 4])
                       for i in range(4)) < 1e-10
            action = (policy.act(scenario, theta) + shift) % 4
            prepared = {'user_id': row['user_id'], 'scenario_id': row['scenario_id'],
                        'permutation_id': shift, 'prompt': row['prompt'],
                        'gold_action': row['gold_action'], 'expected_utilities': row['expected_utilities'],
                        'learned_advice': chr(65 + action), 'policy_checkpoint_sha256': digest,
                        'metrics_by_action': row['metrics_by_action']}
            handle.write(json.dumps(prepared) + '\n')
    print(json.dumps({'event': 'cpu_preparation_complete', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'records': len(rows), 'policy_checkpoint_sha256': digest,
                      'output': str(output)}), flush=True)


def generate(manifest: Path, output_dir: Path, model_name: str, limit: int | None) -> None:
    import torch
    from src.training.model_utils import load_model_with_optional_checkpoint
    from scripts.run_finite_menu_llm_pilot import parse_action

    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    assert len(rows) == 64
    if limit is not None:
        rows = rows[:limit]
    print(json.dumps({'event': 'model_load_start', 'at_utc': datetime.now(timezone.utc).isoformat()}), flush=True)
    model, tokenizer = load_model_with_optional_checkpoint(model_name)
    print(json.dumps({'event': 'model_loaded', 'at_utc': datetime.now(timezone.utc).isoformat(),
                      'gpu_device_count': torch.cuda.device_count(), 'gpu_name': torch.cuda.get_device_name(0),
                      'gpu_memory_allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'base.jsonl').open('w') as handle:
        for i, row in enumerate(rows):
            advice = row['learned_advice']
            msg = [{'role': 'user', 'content': row['prompt']},
                   {'role': 'assistant', 'content': 'TOOL'},
                   {'role': 'user', 'content': f'preference_proxy result: {advice}. '
                    'Final answer: copy that recommended action as one capital letter.'}]
            encoded = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True,
                                                    return_tensors='pt')
            if hasattr(encoded, 'input_ids'):
                encoded = encoded['input_ids']
            encoded = encoded.to(model.device)
            with torch.inference_mode():
                generated = model.generate(encoded, max_new_tokens=24, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            completion = tokenizer.decode(generated[0, encoded.shape[-1]:], skip_special_tokens=True)
            action = parse_action(completion)
            record = {k: v for k, v in row.items() if k != 'metrics_by_action'}
            record.update(messages=msg, rendered_prompt=tokenizer.decode(encoded[0], skip_special_tokens=False),
                          completion=completion, final_action=None if action is None else chr(65 + action),
                          parse_failure=action is None,
                          metrics=None if action is None else row['metrics_by_action'][action],
                          policy_metrics=row['metrics_by_action'][ord(advice) - 65])
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            if i < 6 or (i + 1) % 20 == 0:
                print(json.dumps({'event': 'generation_progress', 'at_utc': datetime.now(timezone.utc).isoformat(),
                                  'records': i + 1, 'advice': advice, 'completion': completion,
                                  'gpu_memory_allocated_bytes': torch.cuda.memory_allocated()}), flush=True)
    (output_dir / 'base_receipt.json').write_text(json.dumps({'model_name': model_name,
        'manifest': str(manifest), 'records': len(rows),
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--source', type=Path, required=True)
    a.add_argument('--checkpoint', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('generate')
    b.add_argument('--manifest', type=Path, required=True)
    b.add_argument('--output-dir', type=Path, required=True)
    b.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct')
    b.add_argument('--limit', type=int)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.source, args.checkpoint, args.output)
    else:
        generate(args.manifest, args.output_dir, args.model_name, args.limit)
