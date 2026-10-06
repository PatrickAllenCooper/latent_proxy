"""One capped CPU staging/import equivalence diagnostic; no model/tokenizer calls."""
import hashlib,importlib,importlib.metadata as md,json,sys,time
from pathlib import Path
from stage_verified_python_sources import stage_packages,verify_staged
from indexed_startup import verify_metadata,indexed_import
root=Path(sys.argv[1]);expected=json.loads(Path(sys.argv[2]).read_text());site=Path(sys.argv[3]);assert not root.exists()
def mark(stage,**kw):print(json.dumps({'stage':stage,'at':time.time(),**kw}),flush=True)
mark('start');verify_metadata(expected);mark('metadata_valid')
mark('staging_start');receipt=stage_packages(site,root,('torch','sympy','accelerate'));mark('staging_complete',files=len(receipt['files']))
assert verify_staged(receipt);mark('source_identity_valid')
# Recompute mapping through original algorithm/index; no cached availability fabricated.
sys.path.insert(0,str(root));verify_metadata(expected)
def imports():return [importlib.import_module(name) for name in ('torch','sympy','accelerate')]
mark('staged_import_start');modules,captures=indexed_import(imports);mark('staged_import_complete')
for module in modules:
 assert Path(module.__file__).resolve().is_relative_to(root.resolve()),module.__file__
 name=module.__name__;assert module.__version__==md.version(name)
assert verify_staged(receipt);verify_metadata(expected)
result={'complete':True,'source_bytes_verified':True,'native_links_verified':True,'versions_match_metadata':True,'metadata_unchanged':True,'module_origins_staged':True,'captures':captures,'model_calls':0,'tokenizer_calls':0,'import_equivalence_scope':'Original declared versions and exact content/resource links, no numerical computation or full model behavior verified','full_environment_identity':False,'qualification_ready':False}
(root/'integration_receipt.json').write_text(json.dumps(result,indent=2)+'\n');mark('complete')
