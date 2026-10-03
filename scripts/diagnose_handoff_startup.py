"""Same inference runner, explicit verified snapshot, bounded loading and stacks."""
import faulthandler,json,os,signal,sys,time
from pathlib import Path
faulthandler.enable()
receipt=json.loads(Path(os.environ['CACHE_RECEIPT']).read_text());assert receipt['valid']
print(json.dumps({'event':'imports_start','timestamp':time.time()}),flush=True)
import src.training.model_utils as model_utils
original=model_utils.load_model_with_optional_checkpoint

def timeout(signum,frame):
    faulthandler.dump_traceback(all_threads=True)
    raise TimeoutError('90-second model/tokenizer startup limit exceeded')

def bounded(*args,**kwargs):
    signal.signal(signal.SIGALRM,timeout);signal.alarm(90)
    print(json.dumps({'event':'bounded_load_start','snapshot':receipt['snapshot'],'timestamp':time.time()}),flush=True)
    try:return original(*args,**kwargs)
    finally:signal.alarm(0)
model_utils.load_model_with_optional_checkpoint=bounded
from scripts.run_learned_proxy_handoff import generate
print(json.dumps({'event':'imports_complete','timestamp':time.time()}),flush=True)
generate(Path(os.environ['MANIFEST']),Path(os.environ['OUTPUT_DIR']),receipt['snapshot'],4)
