"""Bounded CPU integration only; no Transformers/tokenizer/model imports."""
import hashlib,importlib.metadata as md,json,os,sys,time
from pathlib import Path
from indexed_metadata_exists import capture_original_mapping
root=Path(sys.argv[1]);assert not root.exists();root.mkdir()
def mark(stage,**kw):print(json.dumps({'stage':stage,'at':time.time(),**kw}),flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def environment_binding():
    rows=[]
    for d in md.distributions():
        path=Path(d._path);files={}
        for name in ('METADATA','PKG-INFO','RECORD','top_level.txt','entry_points.txt','direct_url.json'):
            p=path/name
            if p.is_file():files[name]=sha(p)
        rows.append({'path':str(path.resolve()),'name':d.metadata['Name'],'version':d.version,'metadata_hashes':files})
    identity={'sys_path':sys.path,'python':sys.version,'python_sha256':sha(sys.executable),'metadata_source_sha256':sha(md.__file__),'distributions':sorted(rows,key=lambda r:r['path'])}
    return identity,hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
mark('start',pid=os.getpid());original_exists=Path.exists
before,digest_before=environment_binding();(root/'environment_before.json').write_text(json.dumps(before,indent=2)+'\n');mark('binding_complete',digest=digest_before)
mark('indexed_mapping_start');indexed,index_receipt=capture_original_mapping(md);assert Path.exists is original_exists
(root/'indexed_mapping.json').write_text(json.dumps(indexed,sort_keys=True)+'\n');(root/'index_receipt.json').write_text(json.dumps(index_receipt,indent=2)+'\n');mark('indexed_mapping_complete',packages=len(indexed),directories=index_receipt['indexed_directories'],lookups=index_receipt['lookups'],restored=True)
mark('unmodified_original_mapping_start');baseline=md.packages_distributions();mark('unmodified_original_mapping_complete',packages=len(baseline));assert indexed==baseline,'mapping differs'
after,digest_after=environment_binding();assert digest_after==digest_before,'environment metadata changed'
versions={r['name']:r['version'] for r in after['distributions']}
for distributions in indexed.values():
 for name in distributions:assert md.version(name)==versions[name]
receipt={'complete':True,'mapping_equal':True,'version_associations_equal':True,'metadata_binding_equal':True,'environment_binding_sha256':digest_before,'Path_exists_restored':Path.exists is original_exists,'identity_scope':'Python executable/source/sys.path/distribution metadata; does not cryptographically hash all package contents or prove environment immutability','runtime_deployment_qualified':False,'model_calls':0,'GPU_jobs':0}
(root/'integration_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');mark('complete')
