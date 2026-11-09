"""Counterfactual-prior cascade (on-path histories only) (verdicts_prior60): do agents follow the exact Bayes rule under a 60/40 prior, or the
textbook 50/50 BHW rule ("override at an informative lead of two")?

Exact rule with prior pi = logit(0.6) and lambda = logit(0.65): public log-odds L starts at pi; agent k says COMPLIANT
iff L + lambda * s_k > 0 (no ties, since pi is not a multiple of lambda). A verdict reveals the signal only while
|L| < lambda, and then L moves by +-lambda. So one revealed COMPLIANT verdict already makes later agents ignore a FAIL
(override threshold 1 toward COMPLIANT), while overriding a PASS toward NON-COMPLIANT still needs two revealed FAILs.
The textbook rule ignores the prior: override iff the informative lead (threshold 2, ties to own signal) is >= 2
against the signal. Decisions where the two rules disagree are the test.

  uv run --isolated --no-project --with numpy python tools/cf_analyse.py   -> prints, writes r3-herding/cf/summary.json
"""
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r3_herding as h  # noqa: E402
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

B = pathlib.Path(SWARM_ROOT + "/r3-herding/cf")
PRIOR = 0.60
PI, LAM = math.log(PRIOR / (1 - PRIOR)), math.log(h.P / (1 - h.P))
LABEL = {("openai/gpt-6.1-sol", "think"): "GPT-6.1", ("claude-sonnet-5-5", "think"): "Sonnet 5.5",
         ("claude-opus-5-5", "think"): "Opus 5.5", ("claude-haiku-5-5", "fast"): "Haiku 5.5 (fast)",
         ("qwen/qwen3-14b", "think"): "Qwen3-14B (reasoning)"}


def sig(x):
    return 1 / (1 + math.exp(-x))


def exact(signal, prev):
    """Bayes action and P(compliant) under the prior, treating predecessors as rational (verdicts reveal signals only
    while |L| < lambda)."""
    L, onpath = PI, True
    for v in prev:
        if abs(L) < LAM:
            L += LAM if v == 1 else -LAM
        elif v != int(L > 0):
            onpath = False            # a rational predecessor in a cascade cannot announce this: Bayes undefined
    s = 1 if signal == 1 else -1
    post = L + LAM * s
    return int(post > 0), sig(post), onpath


def textbook(signal, prev):
    """50/50 BHW rule: informative lead (threshold 2), ties broken toward the own signal."""
    d = h.bhw_public(prev)[len(prev)]
    s = 1 if signal == 1 else -1
    net = s + d
    return signal if net == 0 else int(net > 0)


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


def main():
    recs = {}
    for line in open(B / "calls.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok") and r["cond"] == "verdicts_prior60":
            recs[r["key"]] = r
    out = {"prior": PRIOR, "models": {}}
    for (mid, mode), lab in LABEL.items():
        rs = [r for r in recs.values() if r["model"] == mid and r["mode"] == mode]
        if not rs:
            continue
        # complete chains only
        by_ep = {}
        for r in rs:
            by_ep.setdefault(r["ep"], []).append(r)
        eps = [e for e, v in by_ep.items() if len(v) == h.N_AGENTS]
        disc_bayes, disc_g, agree_bayes, agree_g, perr, perr_g, toC, toC_g, toN, toN_g = ([] for _ in range(10))
        for e in eps:
            for r in by_ep[e]:
                ex_a, ex_p, onpath = exact(r["signal"], r["prev_verdicts"])
                if not onpath:
                    continue
                tb_a = textbook(r["signal"], r["prev_verdicts"])
                agree_bayes.append(float(r["verdict"] == ex_a))
                agree_g.append(e)
                if r["p_compliant"] is not None:
                    perr.append(abs(r["p_compliant"] / 100 - ex_p))
                    perr_g.append(e)
                if ex_a != tb_a:
                    disc_bayes.append(float(r["verdict"] == ex_a))
                    disc_g.append(e)
                    # direction of the discriminating case
                    if ex_a == 1:
                        toC.append(float(r["verdict"] == 1))
                        toC_g.append(e)
                    else:
                        toN.append(float(r["verdict"] == 0))
                        toN_g.append(e)
        m = {"episodes": len(eps), "decisions": len(agree_bayes),
             "agree_exact_all": boot(agree_bayes, agree_g),
             "discriminating_follow_exact": boot(disc_bayes, disc_g),
             "disc_exact_says_C": boot(toC, toC_g), "disc_exact_says_N": boot(toN, toN_g),
             "p_err_exact": boot(perr, perr_g)}
        out["models"][lab] = m
        f = lambda v: "NA" if not v else f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}] n={v[3]}"  # noqa: E731
        print(f"{lab:24s} eps={len(eps):3d} | agree exact (all) {f(m['agree_exact_all'])} | DISCRIMINATING: follows "
              f"exact {f(m['discriminating_follow_exact'])} (exact=C {f(m['disc_exact_says_C'])}; exact=N "
              f"{f(m['disc_exact_says_N'])}) | |P-Pexact| {f(m['p_err_exact'])}")
    (B / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {B / 'summary.json'}")


if __name__ == "__main__":
    main()
