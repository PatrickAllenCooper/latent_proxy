"""Paired measurement of discovery and decision losses from the saved bridge."""
import csv, json
from pathlib import Path
import numpy as np

root=Path('outputs/preference_program')
protocol=json.loads((root/'manifests/discovery_loss_attribution_v33.json').read_text())
rows=list(csv.DictReader((root/'results/discovery_proxy_bridge_v32/per_user.csv').open()))
lookup={(r['arm'],int(r['budget']),r['model'],int(r['user_id'])):float(r['normalized_regret']) for r in rows}
idx=np.random.default_rng(33001).integers(0,200,size=(10000,200))
def stats(v):
    return {'mean':float(v.mean()),'paired_user_bootstrap_95':np.quantile(v[idx].mean(axis=1),[.025,.975]).tolist()}
def values(b,m):return np.array([lookup['eig',b,m,u] for u in range(200)])
report={'protocol':protocol,'budgets':{},'eight_minus_zero':{},'validation':{'unique_user_cells':len(lookup),'expected_user_cells':12800}}
assert len(lookup)==12800
for b in protocol['budgets']:
    a=values(b,'natural_imitation'); e=values(b,'point_exact'); p=values(b,'posterior_exact')
    report['budgets'][b]={'natural_minus_point_exact':stats(a-e),'point_minus_posterior_exact':stats(e-p)}
for m in ('frozen','natural_imitation','point_exact','posterior_exact'):
    report['eight_minus_zero'][m]=stats(values(8,m)-values(0,m))
out=root/'results/discovery_loss_attribution_v33';out.mkdir(exist_ok=True)
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
