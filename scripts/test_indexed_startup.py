"""Local startup hook/restoration fixtures, no Transformers import."""
from pathlib import Path
from types import SimpleNamespace
from indexed_startup import indexed_import
original=lambda:{'pkg':['dist']};module=SimpleNamespace(packages_distributions=original)
result,captures=indexed_import(lambda:module.packages_distributions(),module)
assert result=={'pkg':['dist']} and len(captures)==1 and module.packages_distributions is original
try:indexed_import(lambda:(_ for _ in ()).throw(ValueError('fixture')),module)
except ValueError:pass
assert module.packages_distributions is original
print('Local direct-original-map hook and success/failure restoration passed')
