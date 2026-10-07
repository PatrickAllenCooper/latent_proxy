"""Inspect pinned runtime protocol on CPU; no imports/model calls or execution."""
import ast
import hashlib
import json
from pathlib import Path

def check(path):
    source=Path(path).read_text();tree=ast.parse(source)
    functions={n.name:n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    generate=functions['generate'];sample=functions['_sample']
    assert 'streamer' in [a.arg for a in generate.args.args+generate.args.kwonlyargs]
    assert 'streamer' in [a.arg for a in sample.args.args+sample.args.kwonlyargs]
    def calls(f):
        return [ast.unparse(n) for n in ast.walk(f) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
                and n.func.value.id=='streamer']
    gcalls=calls(generate);scalls=calls(sample)
    assert 'streamer.put(input_ids.cpu())' in gcalls
    assert 'streamer.put(next_tokens.cpu())' in scalls
    assert 'streamer.end()' in scalls
    assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.argmax' for n in ast.walk(sample))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='isinstance'
                   and any(isinstance(a,ast.Name) and a.id=='streamer' for a in n.args) for n in ast.walk(generate))
    return {'valid':True,'scope':'pinned-source AST protocol plus separate duck-protocol CPU fixtures; not live CUDA integration',
            'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'generate_streamer_calls':gcalls,'sample_streamer_calls':scalls,
            'GPU_calls':0,'model_loads':0,'model_responses':0,
            'timing_risk':'Runtime .cpu() token delivery can synchronize GPU and perturb latency.'}

if __name__=='__main__':
    import sys
    print(json.dumps(check(sys.argv[1]),indent=2))
