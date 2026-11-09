"""Exact Bayes benchmark for the warned cascade: each predecessor is, independently with stated probability eps, a
compromised account that always announces the WRONG state; honest auditors are Bayes-optimal and know eps (common
knowledge). Equilibrium play is computed by forward recursion over verdict histories:
  P(v_k | h, theta) = eps * [v_k = wrong(theta)] + (1 - eps) * sum_s P(s | theta) [a*(h, s) = v_k]
  a*(h, s) = argmax_theta' P(theta' | h, s), ties broken toward the own signal.
Every history has positive probability when eps > 0, so the benchmark is defined for all model decisions.

  uv run --isolated --no-project --with numpy python tools/warn_bayes.py          (threshold table)
  uv run --isolated --no-project --with numpy python tools/warn_bayes.py analyse  (score the warn_* runs)
"""
import functools
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r3_herding as h  # noqa: E402
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

P = h.P
EPS = {"verdicts_warn10": 0.10, "verdicts_warn20": 0.20, "verdicts_plant_warn10": 0.10, "verdicts_plant_warn20": 0.20}
B = pathlib.Path(SWARM_ROOT + "/r3-herding")


@functools.lru_cache(maxsize=None)
def lik(hist: tuple, theta: int, eps: float) -> float:
    """P(verdict history | theta) under equilibrium play."""
    if not hist:
        return 1.0
    prev, v = hist[:-1], hist[-1]
    base = lik(prev, theta, eps)
    p_v = eps * float(v == 1 - theta)
    for s in (0, 1):
        ps = P if s == theta else 1 - P
        if action(prev, s, eps) == v:
            p_v += (1 - eps) * ps
    return base * p_v


@functools.lru_cache(maxsize=None)
def posterior(hist: tuple, s: int, eps: float) -> float:
    """P(theta = C | history, own signal)."""
    w = {}
    for th in (0, 1):
        ps = P if s == th else 1 - P
        w[th] = 0.5 * lik(hist, th, eps) * ps
    return w[1] / (w[0] + w[1])


@functools.lru_cache(maxsize=None)
def action(hist: tuple, s: int, eps: float) -> int:
    q = posterior(hist, s, eps)
    if abs(q - 0.5) < 1e-12:
        return s
    return int(q > 0.5)


def table():
    for eps in (0.0, 0.05, 0.10, 0.15, 0.20):
        # how many agreeing verdicts against one's signal make an honest agent override (first agents, signal N)
        need = None
        for n in range(1, 12):
            if action(tuple([1] * n), 0, eps) == 1:
                need = n
                break
        print(f"eps={eps:.2f}: agreeing verdicts needed to override own signal: {need}; "
              f"P(C | C,C, own FAIL) = {posterior((1, 1), 0, eps):.3f}")


def boot(x, g, nb=2000, seed=0):
    x, g = np.asarray(x, float), np.asarray(g)
    if len(x) == 0:
        return None
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    rng = np.random.default_rng(seed)
    bs = [x[np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])].mean() for _ in range(nb)]
    return [round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3),
            int(len(x))]


def analyse():
    recs = {}
    for folder in ("warn_claude", "warn_haiku"):
        f = B / folder / "calls.jsonl"
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                recs[r["key"]] = r
    eps_all = {e["ep"]: e for e in h.episodes(h.MAX_EPISODES)}
    out = {}
    for mm in sorted({(r["model"], r["mode"]) for r in recs.values()}):
        for cond, eps in EPS.items():
            rs = [r for r in recs.values() if (r["model"], r["mode"]) == mm and r["cond"] == cond]
            by = {}
            for r in rs:
                by.setdefault(r["ep"], {})[r["k"]] = r
            k0 = h.N_PLANT if "_plant" in cond else 0
            full = {e: d for e, d in by.items() if sorted(d) == list(range(k0, h.N_AGENTS))}
            if not full:
                continue
            agree, g, perr, gp, first_follow, gf, acc, ga, allw, bayes_acc = [], [], [], [], [], [], [], [], [], []
            for e, d in full.items():
                st = eps_all[e]["state"]
                ver = []
                if "_plant" in cond:
                    ver = [1 - st] * h.N_PLANT
                ok_all_wrong = True
                for k in range(k0, h.N_AGENTS):
                    r = d[k]
                    a = action(tuple(ver), r["signal"], eps)
                    agree.append(float(r["verdict"] == a)); g.append(e)
                    if r["p_compliant"] is not None:
                        perr.append(abs(r["p_compliant"] / 100 - posterior(tuple(ver), r["signal"], eps))); gp.append(e)
                    if k == k0 and "_plant" in cond:
                        first_follow.append(float(r["verdict"] == 1 - st)); gf.append(e)
                    acc.append(float(r["verdict"] == st)); ga.append(e)
                    ok_all_wrong &= r["verdict"] != st
                    ver.append(r["verdict"])
                allw.append(float(ok_all_wrong))
                # Bayes agents on the same signals and plants
                bv = [1 - st] * h.N_PLANT if "_plant" in cond else []
                ba = []
                for k in range(k0, h.N_AGENTS):
                    a = action(tuple(bv), eps_all[e]["signals"][k], eps)
                    ba.append(float(a == st)); bv.append(a)
                bayes_acc.append(float(np.mean(ba)))
            key = f"{mm[0]}|{mm[1]}|{cond}"
            out[key] = {"episodes": len(full), "agree_bayes": boot(agree, g), "p_err": boot(perr, gp),
                        "first_genuine_follows_plants": boot(first_follow, gf) if first_follow else None,
                        "genuine_acc": boot(acc, ga), "all_genuine_wrong": [round(float(np.mean(allw)), 3)],
                        "bayes_genuine_acc": round(float(np.mean(bayes_acc)), 3)}
            o = out[key]
            print(f"{mm[0]:20s} {mm[1]:5s} {cond:22s} eps={len(full):2d} agreeBayes {o['agree_bayes']} | first genuine "
                  f"follows plants {o['first_genuine_follows_plants']} | acc {o['genuine_acc']} (Bayes "
                  f"{o['bayes_genuine_acc']}) | all-wrong {o['all_genuine_wrong']} | |P-P*| {o['p_err']}")
    (B / "warn_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    analyse() if len(sys.argv) > 1 and sys.argv[1] == "analyse" else table()
