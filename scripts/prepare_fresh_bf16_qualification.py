"""Freeze CPU-only prospective numeric-selection qualification. No model imports."""
import hashlib, json, random
from collections import Counter
from pathlib import Path

ROOT = Path('outputs/preference_program/manifests')
SEED = 430044
rng = random.Random(SEED)
rows = []
seen = set()
for stratum, bounds in [('single_digit', (0, 10)), ('two_digit', (10, 100)), ('three_digit', (100, 1000)), ('four_digit', (1000, 10000))]:
    labels = list('ABCD')
    rng.shuffle(labels)
    for gold in labels:
        values = rng.sample(range(*bounds), 4)
        while tuple(sorted(values)) in seen or set(values) == {1, 2, 3, 4}:
            values = rng.sample(range(*bounds), 4)
        seen.add(tuple(sorted(values)))
        maximum = max(values)
        i = values.index(maximum)
        target = 'ABCD'.index(gold)
        values[i], values[target] = values[target], values[i]
        prompt = 'Numbers: ' + ', '.join(f'{label}={value}' for label, value in zip('ABCD', values)) + '. Which label has the largest number? Reply with exactly one capital letter: A, B, C, or D.'
        assert 'ABCD'[max(range(4), key=lambda j: values[j])] == gold
        assert sum(v == maximum for v in values) == 1
        rows.append({'case_id': f'fresh-v44-{len(rows):02d}', 'stratum': stratum, 'displayed_values': values, 'gold_action': gold, 'prompt': prompt, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'messages': [{'role': 'user', 'content': prompt}]})
assert len(rows) == 16 and Counter(r['gold_action'] for r in rows) == dict.fromkeys('ABCD', 4)
assert len({tuple(sorted(r['displayed_values'])) for r in rows}) == 16
manifest = ROOT / 'fresh_bf16_v44.jsonl'
manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
protocol = {
    'status': 'frozen CPU preparation only; execution not authorized by this request',
    'purpose': 'Numeric-selection measurement qualification, not preference discovery or advice correction',
    'source_result_commit': 'f9ae961', 'seed': SEED, 'cases': 16, 'responses': 16,
    'sampling': 'Distinct numeric multisets sampled without replacement within each array; four magnitude strata, each balanced across gold labels. Arrays are separate cases, not rotations of one array. Forced label balancing means cases are not an iid population sample.',
    'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'snapshot': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
    'weights_sha256': 'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee',
    'precision': 'BF16 unquantized; all parameters BF16 on cuda:0; effective SDPA; use_cache=False; eval mode',
    'prompt': 'Exact numbers/label template used in v43; single user native chat template, no extra system message',
    'decode': {'do_sample': False, 'max_new_tokens': 24, 'max_prompt_tokens': 512},
    'resources_if_separately_executed': {'account': 'ucb736_asc1', 'partition': 'ah200', 'qos': 'gpu-normal', 'gres': 'gpu:h200_2g.35gb:1', 'cpus': 4, 'host_memory': '32G', 'walltime': '00:05:00', 'complete_wall_seconds': 285, 'runtime_stage_seconds': 45, 'load_seconds': 90, 'generation_seconds': 10, 'framework_memory_stop_GiB': 12},
    'resource_evidence': 'v43 BF16 four responses completed in under 0.1s generation; peak allocated 3,127,658,496 bytes; full sequential run 27s. Sixteen responses bounded by ten-second total generation watchdog; completion not guaranteed.',
    'scoring': {'parser': r'\s*([ABCD])\s*', 'oracle': 'Unique displayed integer argmax; strict fullmatch; parse failures counted incorrect', 'pass': '16/16 correct, zero parse failures, zero length caps, EOS observed for every response, complete provenance and resource validation'},
    'logging': ['raw messages and rendered prompt', 'prompt and generated token IDs', 'effective EOS IDs and last token', 'EOS and length caps', 'checkpoint/runtime/runner/manifest hashes', 'parameter dtypes and absence of quantization', 'effective attention/template/decode', 'CPU preparation, staging, loading, generation, cleanup and final timestamps', 'CUDA memory and token progress; MIG telemetry uncertainty'],
    'stopping_rule': 'One frozen attempt only. No prompt/model/threshold changes based on qualification outputs; no retry or enlargement. Any scientific failure stops for conceptual assay review. Incomplete or provenance/resource failure yields no qualification. Passing permits only a separately registered next verification gate, not a 320-response study.',
    'limitations': 'v43 was one reused case with fixed NF4-first order and precision-specific kernels/loading paths. This BF16-only gate cannot resolve those causal confounds. Positive integer selection does not establish arithmetic, decimals, preference learning, proxy use, or general human assistance.',
    'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(ROOT / 'fresh_bf16_v44_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
(ROOT / 'fresh_bf16_v44_validation.json').write_text(json.dumps({'CPU_only': True, 'model_calls': 0, 'GPU_allocations': 0, 'unique_cases': 16, 'unique_numeric_multisets': 16, 'gold_counts': dict(Counter(r['gold_action'] for r in rows)), 'unique_maxima_verified': True, 'oracle_and_prompt_verified': True, 'reused_case_excluded': True, 'manifest_sha256': protocol['manifest_sha256'], 'protocol_sha256': hashlib.sha256((ROOT / 'fresh_bf16_v44_protocol.json').read_bytes()).hexdigest()}, indent=2) + '\n')
print('Frozen 16 fresh cases; CPU validation passed; no inference or submission.')
