"""Local CPU real-filesystem equivalence fixtures; no packages/model imports."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
from indexed_metadata_exists import ExistsIndex,DirectoryChanged,capture_original_mapping
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);(root/'file').write_text('x');(root/'directory').mkdir();(root/'live').symlink_to(root/'file');(root/'dangling').symlink_to(root/'absent')
    paths=[root/'file',root/'directory',root/'absent',root/'live',root/'dangling',root/'missing'/'child',root,root/'directory'/'..'/'file']
    index=ExistsIndex()
    for p in paths:assert index.exists(p)==p.exists(),str(p)
    index.validate();assert index.exists(root/'absent') is False
    (root/'newfile').write_text('mutation')
    try:index.validate()
    except DirectoryChanged:pass
    else:raise AssertionError('directory mutation not rejected')
    original=Path.exists
    def mapping():
        return {'fixture':['dist']} if (root/'file').exists() else {}
    result,receipt=capture_original_mapping(SimpleNamespace(packages_distributions=mapping));assert result=={'fixture':['dist']} and Path.exists is original
    def failure():raise RuntimeError('fixture')
    try:capture_original_mapping(SimpleNamespace(packages_distributions=failure))
    except RuntimeError:pass
    assert Path.exists is original
print(json.dumps({'real_filesystem_equivalence_cases':len(paths),'mutation_rejected':True,'mapping_and_restoration_verified':True,'exception_restoration_verified':True,'model_calls':0,'remote_attempts':0}))
# Exercise the actual standard-library algorithm with isolated real metadata.
import importlib.metadata as md
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);info=root/'fixture_dist-1.0.dist-info';info.mkdir()
    (info/'METADATA').write_text('Metadata-Version: 2.1\nName: fixture-dist\nVersion: 1.0\n')
    (root/'fixture_pkg').mkdir();(root/'fixture_pkg'/'__init__.py').write_text('')
    (info/'RECORD').write_text('fixture_pkg/__init__.py,,\nmissing_pkg/__init__.py,,\nfixture_dist-1.0.dist-info/METADATA,,\n')
    dist=md.PathDistribution(info);original_distributions=md.distributions
    md.distributions=lambda:iter([dist])
    try:
        baseline=md.packages_distributions()
        accelerated,_=capture_original_mapping(md)
        assert accelerated==baseline
    finally:md.distributions=original_distributions
print('Actual stdlib mapping equivalence fixture passed; missing RECORD path retained original semantics')
