"""Review-only exact metadata cache loader; never fabricates package availability.
Cache producer must call original packages_distributions in the same pinned environment.
Deployment/production cache creation has not been authorized or performed.
"""
import hashlib,json
from pathlib import Path

def load_verified_map(path, environment_digest, metadata_source_digest):
    receipt=json.loads(Path(path).read_text())
    if receipt.get('producer')!='original_importlib.metadata.packages_distributions':
        raise ValueError('non-authoritative mapping producer')
    if receipt.get('environment_digest')!=environment_digest or receipt.get('metadata_source_digest')!=metadata_source_digest:
        raise ValueError('stale environment/source binding')
    mapping=receipt.get('mapping')
    if not isinstance(mapping,dict) or any(not isinstance(k,str) or not isinstance(v,list) or any(not isinstance(x,str) for x in v) for k,v in mapping.items()):
        raise ValueError('invalid mapping')
    digest=hashlib.sha256(json.dumps(mapping,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=receipt.get('mapping_sha256'):raise ValueError('mapping digest mismatch')
    return {k:list(v) for k,v in mapping.items()}

def import_with_verified_map(import_callable, metadata_module, mapping):
    """Temporarily supply exact cached mapping for startup, restore even on failure.
    Does not override version(), find_spec() or invent installed packages.
    Single-threaded startup only; not safe to deploy during concurrent imports.
    """
    original=metadata_module.packages_distributions
    metadata_module.packages_distributions=lambda:{k:list(v) for k,v in mapping.items()}
    try:return import_callable()
    finally:metadata_module.packages_distributions=original
