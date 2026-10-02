"""Compare greedy generation with first-token four-letter logit scoring."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_finite_menu_llm_pilot import generate, parse_action


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
    candidate_ids = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in 'ABCD'}
    print(json.dumps({'event': 'candidate_tokenization', 'ids': candidate_ids}), flush=True)
    if any(len(ids) != 1 for ids in candidate_ids.values()):
        raise RuntimeError('Letter candidates are not single tokens at this answer boundary')
    ids = torch.tensor([candidate_ids[letter][0] for letter in 'ABCD'], device=model.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'base.jsonl').open('w') as handle:
        for i, prepared in enumerate(rows):
            prompt = prepared['prompt']
            encoded = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors='pt')
            if hasattr(encoded, 'input_ids'):
                encoded = encoded['input_ids']
            encoded = encoded.to(model.device)
            with torch.inference_mode():
                logits = model(input_ids=encoded, use_cache=False).logits[0, -1].index_select(0, ids)
                probs = torch.softmax(logits.float(), dim=0)
            values = [float(x) for x in logits.float().cpu().tolist()]
            probabilities = [float(x) for x in probs.cpu().tolist()]
            forced = chr(65 + max(range(4), key=lambda a: values[a]))
            completion = generate(model, tokenizer, prompt, 24)
            greedy = parse_action(completion)
            record = {k: v for k, v in prepared.items() if k != 'metrics_by_action'}
            record.update(condition='base', completion=completion, greedy_parse_failure=greedy is None,
                          greedy_action=None if greedy is None else chr(65 + greedy),
                          forced_action=forced,
                          candidate_logits={letter: values[a] for a, letter in enumerate('ABCD')},
                          candidate_probabilities={letter: probabilities[a] for a, letter in enumerate('ABCD')},
                          greedy_metrics=None if greedy is None else prepared['metrics_by_action'][greedy],
                          forced_metrics=prepared['metrics_by_action'][ord(forced) - 65])
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            if i == 0:
                print(json.dumps({'event': 'first_score', 'at_utc': datetime.now(timezone.utc).isoformat(),
                                  'input_tokens': int(encoded.shape[-1]),
                                  'gpu_memory_allocated_bytes': torch.cuda.memory_allocated(),
                                  'gpu_memory_peak_bytes': torch.cuda.max_memory_allocated(),
                                  'forced_action': forced, 'completion': completion}), flush=True)
            if i < 6 or (i + 1) % 30 == 0:
                print(json.dumps({'event': 'scoring_progress', 'at_utc': datetime.now(timezone.utc).isoformat(),
                                  'records': i + 1, 'arm': prepared['arm'],
                                  'forced_action': forced, 'greedy_completion': completion}), flush=True)
    (output_dir / 'base_receipt.json').write_text(json.dumps({
        'model_name': model_name, 'manifest': str(manifest), 'records': len(rows),
        'candidate_token_ids': candidate_ids,
        'completed_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct')
    args = parser.parse_args()
    run(args.manifest, args.output_dir, args.model_name)
