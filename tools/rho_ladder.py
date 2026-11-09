"""Spearman correlation across models between agreement with the Bayes action on honest verdict chains and the
share of episodes in which two planted verdicts make all genuine agents wrong (models with >= 10 complete planted
episodes and >= 100 honest decisions), with a bootstrap CI over models (5000 resamples, seed 0).
  uv run --isolated --no-project --with numpy --with scipy python tools/rho_ladder.py
"""
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

S = json.load(open(pathlib.Path(SWARM_ROOT + "/r3-herding/"
                                "paper_summary.json"), encoding="utf-8"))
rows = []
for m, d in S.items():
    if m.startswith("_") or "|" in m:
        continue
    x = (d.get("verdicts") or {}).get("bayes_action_agree")
    y = (S["_planted"].get(m, {}).get("verdicts_plant") or {}).get("all_genuine_wrong")
    if x and y and y[3] >= 10 and x[3] >= 100:
        rows.append((m, x[0], y[0]))
x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows])
rho, p = spearmanr(x, y)
rng = np.random.default_rng(0); bs = []
for _ in range(5000):
    i = rng.integers(0, len(x), len(x))
    if len(set(x[i])) > 1 and len(set(y[i])) > 1:
        bs.append(spearmanr(x[i], y[i])[0])
print(f"N={len(rows)} rho={rho:.3f} p={p:.4f} 95% CI {np.percentile(bs, 2.5):.2f}-{np.percentile(bs, 97.5):.2f}")
for r in sorted(rows, key=lambda r: -r[1]):
    print(f"  {r[0]:24s} {r[1]:.3f} {r[2]:.3f}")
