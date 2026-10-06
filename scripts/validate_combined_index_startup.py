"""Production-path CPU startup proposal: no cached mapping, no model/tokenization.
Run only in a single-threaded isolated process with pinned metadata verification.
"""
import hashlib,importlib.metadata as md,json,sys,time,faulthandler
from pathlib import Path
from types import SimpleNamespace
from indexed_metadata_exists import capture_original_mapping
from indexed_source_inspection import inspect_with_index
import os,genericpath

def verify_metadata(expected):
    if sys.version!=expected['python']:raise ValueError('Python version changed')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if sha(sys.executable)!=expected['python_sha256'] or sha(md.__file__)!=expected['metadata_source_sha256']:raise ValueError('Python/source identity changed')
    actual=[]
    for d in md.distributions():
        path=Path(d._path);files={}
        for name in ('METADATA','PKG-INFO','RECORD','top_level.txt','entry_points.txt','direct_url.json'):
            p=path/name
            if p.is_file():files[name]=sha(p)
        actual.append({'path':str(path.resolve()),'name':d.metadata['Name'],'version':d.version,'metadata_hashes':files})
    if sorted(actual,key=lambda r:r['path'])!=expected['distributions']:raise ValueError('Distribution metadata changed')
    return True

def indexed_import(import_callable,metadata_module=md):
    original=metadata_module.packages_distributions;captures=[]
    def fresh():
        print(json.dumps({'stage':'indexed_mapping_hook_start','at':time.time()}),flush=True)
        mapping,receipt=capture_original_mapping(SimpleNamespace(packages_distributions=original))
        print(json.dumps({'stage':'indexed_mapping_hook_complete','at':time.time(),'directories':receipt['indexed_directories'],'lookups':receipt['lookups']}),flush=True)
        captures.append(receipt);return mapping
    metadata_module.packages_distributions=fresh
    try:return import_callable(),captures
    finally:metadata_module.packages_distributions=original

if __name__=='__main__':
    expected=json.loads(Path(sys.argv[1]).read_text());output=Path(sys.argv[2]);assert not output.exists()
    def mark(stage,**kw):print(json.dumps({'stage':stage,'at':time.time(),**kw}),flush=True)
    mark('start',sys_path=sys.path);verify_metadata(expected);mark('metadata_before_valid')
    original=md.packages_distributions;original_exists=Path.exists;original_os=os.path.exists;original_generic=genericpath.exists
    source_paths=[Path(md.__file__),Path('/projects/paco0228/software/anaconda/envs/latent-proxy-env/lib/python3.12/inspect.py'),Path('/projects/paco0228/software/anaconda/envs/latent-proxy-env/lib/python3.12/site-packages/torch/_library/utils.py'),Path('/projects/paco0228/software/anaconda/envs/latent-proxy-env/lib/python3.12/site-packages/torch/utils/_debug_mode/_mode.py')]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    def import_tokenizer_class():
        from transformers import AutoTokenizer
        return AutoTokenizer
    faulthandler.enable();faulthandler.dump_traceback_later(15,repeat=True)
    mark('indexed_AutoTokenizer_import_start');(cls,captures),inspection=inspect_with_index(lambda:indexed_import(import_tokenizer_class));mark('indexed_AutoTokenizer_import_complete',inspection_directories=inspection['indexed_directories'],inspection_lookups=inspection['lookups'])
    faulthandler.cancel_dump_traceback_later()
    assert md.packages_distributions is original and Path.exists is original_exists
    verify_metadata(expected)
    assert os.path.exists is original_os and genericpath.exists is original_generic
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
    origins={name:getattr(module,'__file__',None) for name,module in sys.modules.items() if name in ('torch','sympy','accelerate','transformers')}
    assert all(path and '/projects/paco0228/software/anaconda/envs/latent-proxy-env/' in path for name,path in origins.items() if name!='transformers')
    receipt={'inspection_receipt':inspection,'source_hashes_before_after':source_hashes,'module_origins':origins,'filenames_unchanged':True,'complete':True,'metadata_before_after_valid':True,'restoration_valid':True,'indexed_captures':captures,'model_calls':0,'tokenizer_loads':0,'chat_template_calls':0,'sys_path':sys.path,'full_package_content_identity':False,'qualification_ready':False,'identity_scope':'Pinned Python executable/version/source and complete distribution metadata againstv47; runtime archive independently verified by staging; package content not fullyhashed'}
    output.write_text(json.dumps(receipt,indent=2)+'\n');mark('complete')
