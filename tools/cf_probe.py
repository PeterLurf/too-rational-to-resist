"""Counterfactual-prior probes: single decisions with constructed predecessor histories where the exact Bayes rule
under an unequal prior and the textbook 50/50 BHW rule ("override at an informative lead of two") disagree.

Prior 0.60 compliant (and the mirror, 0.40), p = 0.65. Cases (prev = earlier verdicts, 1 = COMPLIANT):
  prior .60  D1: prev [C],    own FAIL -> exact C (P=.60), textbook N     (discriminating)
             D2: prev [C, N], own FAIL -> OFF-PATH: a rational auditor #2 cannot say N after a C under this
                 prior, so Bayes is undefined; reported separately, not a test
             K1: prev [N],    own PASS -> both C (P=.60)                  (control)
             K2: prev [N, N], own PASS -> both N (P=.45)                  (control)
  prior .40  mirror image with C and N (and PASS and FAIL) swapped.
The same prompt as the live chains (r3_herding.prompt), positions #2 / #3 of 12, 20 items per case with varied
vendor/item/batch and question polarity.

  python tools/cf_probe.py run --models gptcf sonnetcf opuscf haikucf qwen14thinkcf [--n 20] [--workers 12] [--budget 6]
  python tools/cf_probe.py analyse
"""
import argparse
import concurrent.futures as cf
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r3_common as rc  # noqa: E402
import r3_herding as h  # noqa: E402
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

OUT = pathlib.Path(SWARM_ROOT + "/r3-herding/cf_probe")
LAM = math.log(h.P / (1 - h.P))
CASES = {  # name: (prior, prev verdicts, own signal, discriminating)
    "60_D1": (0.60, [1], 0, True), "60_D2": (0.60, [1, 0], 0, True),
    "60_K1": (0.60, [0], 1, False), "60_K2": (0.60, [0, 0], 1, False),
    "40_D1": (0.40, [0], 1, True), "40_D2": (0.40, [0, 1], 1, True),
    "40_K1": (0.40, [1], 0, False), "40_K2": (0.40, [1, 1], 0, False),
}


def exact(prior, prev, signal):
    L = math.log(prior / (1 - prior))
    for v in prev:
        if abs(L) < LAM:
            L += LAM if v == 1 else -LAM
    post = L + LAM * (1 if signal == 1 else -1)
    return int(post > 0), 1 / (1 + math.exp(-post))


def textbook(prev, signal):
    d = h.bhw_public(prev)[len(prev)]
    net = (1 if signal == 1 else -1) + d
    return signal if net == 0 else int(net > 0)


for _c, (_pr, _pv, _s, _disc) in CASES.items():      # the design is what the docstring says
    assert (exact(_pr, _pv, _s)[0] != textbook(_pv, _s)) == _disc, _c


def items(n):
    return h.episodes(h.MAX_EPISODES)[100:100 + n]       # episodes not used as chain starts in the cf runs


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = rc.Cache(OUT / "calls.jsonl")
    tasks = []
    for alias in a.models:
        mid, mode = h.MODELS[alias][:2]
        for cname, (prior, prev, own, _) in CASES.items():
            for ep in items(a.n):
                kk = f"{mid}|{mode}|{cname}|{ep['ep']}"
                if kk not in cache.d:
                    tasks.append((mid, mode, cname, prior, prev, own, ep, kk))
    print(f"{len(tasks)} probe calls", flush=True)

    def fn(t):
        mid, mode, cname, prior, prev, own, ep, kk = t
        if rc.SPEND["usd"] > a.budget:
            return
        cond = "verdicts_prior60" if prior == 0.60 else "verdicts_prior40"
        pr = h.prompt(ep, cond, len(prev), own, list(prev), [])
        for att in range(3):
            r = rc.call(mid, pr, mode)
            ans = rc.parse_yes_no(r["text"]) if r["error"] is None else None
            verdict = None if ans is None else (int(ans == "YES") if ep["polarity"] == 1 else int(ans == "NO"))
            rec = {"key": kk, "ok": verdict is not None, "model": mid, "mode": mode, "case": cname, "prior": prior,
                   "prev": prev, "signal": own, "ep": ep["ep"], "polarity": ep["polarity"], "verdict": verdict,
                   "p_compliant": h.parse_p(r["text"]), "text": r["text"], "cost": r["cost"], "error": r["error"],
                   "attempt": att}
            cache.add(rec)
            if rec["ok"] or (r["error"] or "").startswith("refusal"):
                break

    rc.run_tasks(tasks, fn, workers=a.workers, budget=a.budget, label="cf-probe")
    print(f"DONE spend ${rc.SPEND['usd']:.2f}", flush=True)


def boot(x, nb=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return [round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3),
            int(len(x))]


def analyse(a):
    recs = {}
    for line in open(OUT / "calls.jsonl", encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            recs[r["key"]] = r
    out = {}
    f = lambda v: "NA" if not v else f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}] n={v[3]}"  # noqa: E731
    for mm in sorted({(r["model"], r["mode"]) for r in recs.values()}):
        rs = [r for r in recs.values() if (r["model"], r["mode"]) == mm]
        row = {}
        for disc in (True, False):
            sel = [r for r in rs if CASES[r["case"]][3] == disc and not r["case"].endswith("D2")]
            follow = [float(r["verdict"] == exact(r["prior"], r["prev"], r["signal"])[0]) for r in sel]
            perr = [abs(r["p_compliant"] / 100 - exact(r["prior"], r["prev"], r["signal"])[1]) for r in sel
                    if r["p_compliant"] is not None]
            row["discriminating" if disc else "control"] = {"follows_exact": boot(follow), "p_err_exact": boot(perr)}
        for pr in (0.60, 0.40):
            sel = [r for r in rs if r["case"].endswith("D1") and r["prior"] == pr]
            row[f"disc_prior{int(pr * 100)}"] = boot([float(r["verdict"] == exact(r["prior"], r["prev"], r["signal"])[0])
                                                    for r in sel])
        off = [r for r in rs if r["case"].endswith("D2")]
        row["offpath_D2_mean_p"] = round(float(np.mean([r["p_compliant"] for r in off if r["p_compliant"] is not None]
                                                       or [np.nan])), 2)
        out[f"{mm[0]}|{mm[1]}"] = row
        print(f"{mm[0]:30s} {mm[1]:5s} DISCRIMINATING follows exact {f(row['discriminating']['follows_exact'])} "
              f"(prior .60 {f(row['disc_prior60'])}; .40 {f(row['disc_prior40'])}) |P-P*| "
              f"{f(row['discriminating']['p_err_exact'])} | CONTROL follows {f(row['control']['follows_exact'])}")
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "analyse"])
    ap.add_argument("--models", nargs="+", default=["gptcf", "sonnetcf", "opuscf", "haikucf", "qwen14thinkcf"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=float, default=6.0)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else analyse(a)


if __name__ == "__main__":
    main()
