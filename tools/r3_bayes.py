"""R3-BAYES: single-shot vignettes where the exact Bayes answer DIFFERS from naive evidence counting.

Reviewer objection to the cascade results (r3_herding): "the task is symmetric and fully specified, so a model that just
counts PASS/FAIL (or verdicts) looks Bayesian". These vignettes break the coincidence. Compliance-audit domain as in
r3_herding (spot check correct w.p. 0.65 unless stated), one call per item, reply ends with P_COMPLIANT and ANSWER.

Families (48 items each, fixed seed, saved to items.json):
  F1 PRIOR     stated prior 70/30 (direction randomised); own check + 0-4 earlier standard checks.
  F2 PRECISION prior 50/50; each check is a "standard spot check (65%)" or a "lab assay (90%)".
  F3 ECHO      prior 50/50; some earlier reports are marked re-measurements of a unit already sampled (no new info).
  F4 CASCADE   only earlier CONCLUSIONS visible (r3_herding verdicts wording); own check opposes the cascade.
For every item: exact Bayes posterior/decision and the naive-count posterior/decision (unweighted PASS minus FAIL
including own; ties -> own signal; prior ignored; echoes counted; every verdict counted as a signal). In F1-F3 ~2/3 of
items are "discriminating" (Bayes decision != naive decision); in F4 decisions coincide and the stated probability is
the discriminator.

  items                       -> writes items.json + prereg.md (idempotent, seed-fixed)
  run --models a b --workers W --budget B [--limit N]   (one process per alias; cache calls_<alias>.jsonl)
  analyse                     -> summary.json, analysis.txt, VERDICT_BAYES lines
DRYRUN=1: random answers, no network, outputs under OUT/dryrun/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r3_common as rc  # noqa: E402
from r3_herding import VENDORS, ITEMS, P, parse_p, bhw_public, posterior  # noqa: E402
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

SEED = 20261108
N_PER_FAMILY = 48
N_AGENTS = 12                       # chain length framing for F4 (as r3_herding)
P_LAB = 0.90
PRIOR_HI = 0.70
L_STD = math.log(P / (1 - P))       # 0.619
L_LAB = math.log(P_LAB / (1 - P_LAB))  # 2.197
L_PRIOR = math.log(PRIOR_HI / (1 - PRIOR_HI))  # 0.847

OUT = pathlib.Path(SWARM_ROOT + "/r3-bayes")
if os.environ.get("DRYRUN") == "1":
    OUT = OUT / "dryrun"

MODELS = {  # alias -> (model id, mode)
    "opus": ("claude-opus-5-5", "think"),
    "sonnet": ("claude-sonnet-5-5", "think"),
    "haikuthink": ("claude-haiku-5-5", "think"),
    "haikufast": ("claude-haiku-5-5", "fast"),
    "gpt": ("openai/gpt-6.1-sol", "think"),
    "gemini": ("google/gemini-4-argon", "think"),
    "qwen14think": ("qwen/qwen3-14b", "think"),
    "qwen14fast": ("qwen/qwen3-14b", "fast"),
    "llama8": ("meta-llama/llama-3.1-8b-instruct", "fast"),
}
FAMILIES = ["F1", "F2", "F3", "F4"]


def sigm(x):
    return 1.0 / (1.0 + math.exp(-x))


def sgn(s):
    return 1 if s == 1 else -1


def sig_lab(s):
    return "PASS" if s == 1 else "FAIL"


def ver_lab(v):
    return "COMPLIANT" if v == 1 else "NON-COMPLIANT"


def naive_eval(own, others):
    """Naive count: unweighted PASS minus FAIL including own; tie -> own. Returns (net, decision, posterior)."""
    net = sgn(own) + sum(sgn(x) for x in others)
    dec = own if net == 0 else int(net > 0)
    return net, dec, posterior(net)


# ------------------------------------------------------------------------------------------------ item generation
def _surface(rng, i, fam):
    return {"id": f"{fam}-{i:02d}", "family": fam, "polarity": int(rng.integers(2)),
            "vendor": VENDORS[int(rng.integers(len(VENDORS)))], "item": ITEMS[int(rng.integers(len(ITEMS)))],
            "batch": f"B-{int(rng.integers(1000, 9999))}"}


def _pick(rng, disc_pool, ctrl_pool, n_disc, n_ctrl, key=None):
    """Sample n_disc discriminating and n_ctrl control configs, cycling (with reshuffle) if a pool is small.
    If key is given, controls are balanced across the key's values."""
    def cyc(pool, n):
        out = []
        while len(out) < n:
            idx = rng.permutation(len(pool))
            out += [pool[j] for j in idx[: n - len(out)]]
        return out
    disc = cyc(disc_pool, n_disc)
    if key is None:
        ctrl = cyc(ctrl_pool, n_ctrl)
    else:
        groups = {}
        for c in ctrl_pool:
            groups.setdefault(key(c), []).append(c)
        ks = sorted(groups)
        per = [n_ctrl // len(ks) + (1 if j < n_ctrl % len(ks) else 0) for j in range(len(ks))]
        ctrl = []
        for k, nn in zip(ks, per):
            ctrl += cyc(groups[k], nn)
    return disc, ctrl


def gen_f1(rng):
    pool_d, pool_c = [], []
    for d in (1, 0):                 # prior favours compliant (1) or non-compliant (0)
        for m in range(5):
            for own in (1, 0):
                for kp in range(m + 1):
                    earlier = [1] * kp + [0] * (m - kp)
                    lo = (L_PRIOR if d else -L_PRIOR) + L_STD * (sgn(own) + kp - (m - kp))
                    bdec = int(lo > 0)
                    nnet, ndec, npost = naive_eval(own, earlier)
                    cfg = {"prior_dir": d, "m": m, "own": own, "earlier": earlier, "bayes_lo": lo, "bayes_dec": bdec,
                           "naive_dec": ndec}
                    (pool_d if bdec != ndec else pool_c).append(cfg)
    n_d = round(N_PER_FAMILY * 2 / 3)
    disc, ctrl = _pick(rng, pool_d, pool_c, n_d, N_PER_FAMILY - n_d,
                       key=lambda c: (c["bayes_dec"] == c["prior_dir"], c["m"] > 0))
    items = []
    for i, cfg in enumerate(disc + ctrl):
        it = _surface(rng, i, "F1")
        earlier = [int(x) for x in rng.permutation(cfg["earlier"])]
        own = cfg["own"]
        it.update({"prior_c": PRIOR_HI if cfg["prior_dir"] else 1 - PRIOR_HI, "own": own, "own_type": "std",
                   "reports": [{"k": j + 1, "type": "std", "result": r, "echo_of": None} for j, r in enumerate(earlier)],
                   "bayes_p": sigm(cfg["bayes_lo"]), "bayes_dec": cfg["bayes_dec"]})
        nnet, ndec, npost = naive_eval(own, earlier)
        it.update({"naive_net": nnet, "naive_dec": ndec, "naive_p": npost, "disc": ndec != it["bayes_dec"]})
        items.append(it)
    return items


def gen_f2(rng):
    pool_d, pool_c = [], []
    for n in range(2, 7):                         # total checks incl. own
        for nl in (1, 2):                         # total lab assays
            ns = n - nl
            if ns < 1:
                continue
            for sp in range(ns + 1):
                for lp in range(nl + 1):
                    sf, lf = ns - sp, nl - lp
                    lo = L_STD * (sp - sf) + L_LAB * (lp - lf)
                    if abs(lo) < 1e-9:
                        continue
                    checks = [("std", 1)] * sp + [("std", 0)] * sf + [("lab", 1)] * lp + [("lab", 0)] * lf
                    own_idxs = {checks[j]: j for j in range(len(checks))}.values()   # one own per distinct (type,result)
                    for own_idx in sorted(own_idxs):
                        own_t, own_r = checks[own_idx]
                        others = checks[:own_idx] + checks[own_idx + 1:]
                        nnet, ndec, npost = naive_eval(own_r, [r for _, r in others])
                        bdec = int(lo > 0)
                        cfg = {"n": n, "nl": nl, "own": (own_t, own_r), "others": others, "bayes_lo": lo,
                               "bayes_dec": bdec, "naive_dec": ndec}
                        (pool_d if bdec != ndec else pool_c).append(cfg)
    # de-duplicate identical (own, multiset others)
    def dd(pool):
        seen, out = set(), []
        for c in pool:
            k = (c["own"], tuple(sorted(c["others"])))
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out
    pool_d, pool_c = dd(pool_d), dd(pool_c)
    n_d = round(N_PER_FAMILY * 2 / 3)
    disc, ctrl = _pick(rng, pool_d, pool_c, n_d, N_PER_FAMILY - n_d, key=lambda c: c["n"])
    items = []
    for i, cfg in enumerate(disc + ctrl):
        it = _surface(rng, i, "F2")
        others = [cfg["others"][j] for j in rng.permutation(len(cfg["others"]))]
        own_t, own_r = cfg["own"]
        it.update({"prior_c": 0.5, "own": own_r, "own_type": own_t,
                   "reports": [{"k": j + 1, "type": t, "result": r, "echo_of": None} for j, (t, r) in enumerate(others)],
                   "bayes_p": sigm(cfg["bayes_lo"]), "bayes_dec": cfg["bayes_dec"]})
        nnet, ndec, npost = naive_eval(own_r, [r for _, r in others])
        it.update({"naive_net": nnet, "naive_dec": ndec, "naive_p": npost, "disc": ndec != it["bayes_dec"]})
        items.append(it)
    return items


def gen_f3(rng):
    pool_d, pool_c = [], []
    for u in range(2, 5):                                  # unique earlier units
        for own in (1, 0):
            for kp in range(u + 1):
                uniq = [1] * kp + [0] * (u - kp)
                lo = L_STD * (sgn(own) + kp - (u - kp))
                if abs(lo) < 1e-9:
                    continue
                # echo multiset: counts per unique earlier index, each 0..2, total 1..3
                import itertools
                for cnt in itertools.product(range(3), repeat=u):
                    e = sum(cnt)
                    if not 1 <= e <= 3:
                        continue
                    echoed = [uniq[j] for j in range(u) for _ in range(cnt[j])]
                    nnet, ndec, npost = naive_eval(own, uniq + echoed)
                    bdec = int(lo > 0)
                    cfg = {"u": u, "own": own, "uniq": uniq, "cnt": cnt, "bayes_lo": lo, "bayes_dec": bdec,
                           "naive_dec": ndec}
                    (pool_d if bdec != ndec else pool_c).append(cfg)
    def dd(pool):   # dedupe on (own, sorted uniq, sorted echoed results) -> keeps distinct result patterns
        seen, out = set(), []
        for c in pool:
            ech = tuple(sorted(c["uniq"][j] for j in range(c["u"]) for _ in range(c["cnt"][j])))
            k = (c["own"], tuple(sorted(c["uniq"])), ech)
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out
    pool_d, pool_c = dd(pool_d), dd(pool_c)
    n_d = round(N_PER_FAMILY * 2 / 3)
    disc, ctrl = _pick(rng, pool_d, pool_c, n_d, N_PER_FAMILY - n_d, key=lambda c: c["u"])
    items = []
    for i, cfg in enumerate(disc + ctrl):
        it = _surface(rng, i, "F3")
        u, own = cfg["u"], cfg["own"]
        order = [int(x) for x in rng.permutation(u)]          # original units in speaking order
        seq = [{"type": "std", "result": cfg["uniq"][j], "echo_of": None, "_orig": j} for j in order]
        for j in range(u):
            for _ in range(cfg["cnt"][j]):
                pos_orig = next(p for p, r in enumerate(seq) if r.get("_orig") == j)
                ins = int(rng.integers(pos_orig + 1, len(seq) + 1))
                seq.insert(ins, {"type": "std", "result": cfg["uniq"][j], "echo_of": j, "_orig": None})
        reports = []
        for k, r in enumerate(seq):
            r["k"] = k + 1
        for r in seq:
            eo = None if r["echo_of"] is None else next(x["k"] for x in seq if x["_orig"] == r["echo_of"])
            reports.append({"k": r["k"], "type": "std", "result": r["result"], "echo_of": eo})
        it.update({"prior_c": 0.5, "own": own, "own_type": "std", "reports": reports,
                   "bayes_p": sigm(cfg["bayes_lo"]), "bayes_dec": cfg["bayes_dec"]})
        nnet, ndec, npost = naive_eval(own, [r["result"] for r in reports])
        it.update({"naive_net": nnet, "naive_dec": ndec, "naive_p": npost, "disc": ndec != it["bayes_dec"]})
        items.append(it)
    return items


def gen_f4(rng):
    pats = []
    for L in range(3, 12):                                  # pure cascade d,d,d,...
        pats.append(("pure", [1] * L))
    for k in range(1, 8):                                   # mixed starts, then k agreeing conclusions
        pats.append(("mixA", [1, 0, 1, 1] + [1] * k))       # lead 0,+1,0,+1,+2 ...
        pats.append(("mixB", [0, 1, 1, 1] + [1] * k))       # lead 0,-1,0,+1,+2 ...
    cfgs = []
    while len(cfgs) < N_PER_FAMILY:
        for j in rng.permutation(len(pats)):
            if len(cfgs) < N_PER_FAMILY:
                cfgs.append(pats[j])
    items = []
    for i, (kind, pat) in enumerate(cfgs):
        d = int(rng.integers(2))                            # cascade direction: 1 compliant, 0 non-compliant
        verdicts = [v if d else 1 - v for v in pat]
        own = 1 - d                                         # own check opposes the cascade
        lead = bhw_public(verdicts)[-1]
        bnet = sgn(own) + lead
        assert bnet != 0
        nnet, ndec, npost = naive_eval(own, verdicts)
        it = _surface(rng, i, "F4")
        it.update({"prior_c": 0.5, "own": own, "own_type": "std", "reports": [], "verdicts": verdicts,
                   "pattern": kind, "bhw_lead": lead, "bayes_net": bnet, "bayes_p": posterior(bnet),
                   "bayes_dec": int(bnet > 0), "naive_net": nnet, "naive_dec": ndec, "naive_p": npost})
        it["disc"] = it["naive_dec"] != it["bayes_dec"]
        items.append(it)
    return items


def build_items():
    rng = np.random.default_rng(SEED)
    items = gen_f1(rng) + gen_f2(rng) + gen_f3(rng) + gen_f4(rng)
    for it in items:
        it["prompt"] = prompt(it)
        it["bayes_p"] = round(it["bayes_p"], 6)
        it["naive_p"] = round(it["naive_p"], 6)
    return items


def load_items():
    p = OUT / "items.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    items = build_items()
    OUT.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=1), encoding="utf-8")
    return items


# ---------------------------------------------------------------------------------------------------------- prompt
def prompt(it: dict) -> str:
    fam = it["family"]
    pc = it["prior_c"]
    if fam == "F4":
        k = len(it["verdicts"])
        head = (f"You are auditor #{k + 1} of {N_AGENTS} in a sequential compliance review of batch {it['batch']} of "
                f"{it['item']} supplied by {it['vendor']}.")
    elif it["reports"]:
        head = (f"You are the last of {len(it['reports']) + 1} auditors in a compliance review of batch {it['batch']} "
                f"of {it['item']} supplied by {it['vendor']}.")
    else:
        head = f"You are the auditor in a compliance review of batch {it['batch']} of {it['item']} supplied by {it['vendor']}."
    lines = [head, "", "Facts known to every auditor:"]
    if abs(pc - 0.5) < 1e-9:
        lines.append("- Before any checks, the batch is equally likely to be compliant or non-compliant (50/50).")
    else:
        lines.append(f"- Before any checks, based on this supplier's record, the batch is compliant with probability "
                     f"{pc:.2f} and non-compliant with probability {1 - pc:.2f}.")
    std = ("If the batch is compliant, a standard spot check comes out PASS with probability "
           f"{P:.2f} and FAIL with probability {1 - P:.2f}. If the batch is non-compliant, it comes out FAIL with "
           f"probability {P:.2f} and PASS with probability {1 - P:.2f}.")
    if fam == "F2":
        lines.append("- Two kinds of test are in use, and each report below states which one was used. " + std +
                     f" A laboratory assay is more accurate: if the batch is compliant it comes out PASS with probability "
                     f"{P_LAB:.2f} and FAIL with probability {1 - P_LAB:.2f}; if the batch is non-compliant it comes out "
                     f"FAIL with probability {P_LAB:.2f} and PASS with probability {1 - P_LAB:.2f}.")
        lines.append("- Each auditor tests one random unit from the batch. Test results are independent of each other "
                     "given the batch's true status.")
    else:
        lines.append("- Each auditor independently samples one random unit from the batch for a private spot check. " + std +
                     " Spot checks of different units are independent of each other given the batch's true status.")
    if fam == "F3":
        lines.append("- Some auditors did not draw a fresh unit but re-measured a unit that an earlier auditor had already "
                     "sampled. The spot-check result for a given unit is deterministic: re-measuring the same unit always "
                     "reproduces the original result for that unit. Re-measurements are marked below.")
    if fam == "F4":
        lines.append("- Auditors report one at a time. All auditors received these same instructions, are equally "
                     "competent, and each sees their own spot check plus the stated conclusions of the auditors before "
                     "them (not their spot checks).")
        lines.append("- Each auditor is scored only on whether their own conclusion matches the batch's true status.")
    else:
        lines.append("- You are scored only on whether your conclusion matches the batch's true status.")
    own_t = "laboratory assay" if it["own_type"] == "lab" else "standard spot check"
    if fam == "F2":
        lines += ["", f"Your private test ({own_t}, on a fresh unit): {sig_lab(it['own'])}."]
    elif fam == "F3":
        lines += ["", f"Your private spot check (fresh unit): {sig_lab(it['own'])}."]
    else:
        lines += ["", f"Your private spot check: {sig_lab(it['own'])}."]
    if fam == "F4":
        lines += ["", "Conclusions stated by earlier auditors, in order:"]
        lines += [f"  Auditor #{j + 1}: {ver_lab(v)}" for j, v in enumerate(it["verdicts"])]
    elif it["reports"]:
        lines += ["", "Spot-check results of the earlier auditors, in order (not their conclusions):" if fam != "F2"
                  else "Test results of the earlier auditors, in order (not their conclusions):"]
        for r in it["reports"]:
            tag = ""
            if fam == "F2":
                tag = " (laboratory assay)" if r["type"] == "lab" else " (standard spot check)"
            elif fam == "F3":
                tag = (f" (re-measured the same physical unit that Auditor #{r['echo_of']} sampled, so this result "
                       f"necessarily repeats #{r['echo_of']}'s)" if r["echo_of"] else " (fresh unit)")
            lines.append(f"  Auditor #{r['k']}: {sig_lab(r['result'])}{tag}")
    else:
        lines += ["", "There are no other auditors and no other evidence."]
    q = "compliant" if it["polarity"] == 1 else "non-compliant"
    lines += ["", f"Question: Is batch {it['batch']} {q}? Give the answer that is more likely to be correct.",
              "End your reply with exactly two lines:",
              "P_COMPLIANT: <your probability, 0-100, that the batch is compliant>",
              "ANSWER: YES or ANSWER: NO"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------- prereg
PREREG = """# Pre-registration: R3-BAYES (Bayes vs naive-count vignettes)

Written before any real API call (DRYRUN smoke test only). Items: 4 families x 48 = 192 single-shot vignettes,
seed {seed}, saved in items.json with exact Bayes posterior/decision and naive-count posterior/decision.

Naive count = unweighted PASS minus FAIL including own signal; ties -> own signal; stated prior ignored; re-measurement
echoes counted; every earlier verdict counted as one signal. Exact Bayes = log-odds sum with the stated prior, stated
test accuracies (0.65 standard, 0.90 lab), each physical unit counted once, and (F4) only BHW-informative verdicts.

Models (one call per item): {models}

Metrics per model x family (95% cluster bootstrap over items, rc.boot): Bayes-agreement and naive-agreement on
discriminating items (Bayes decision != naive decision); accuracy vs Bayes on control items; mean |P/100 - Bayes
posterior| and mean |P/100 - naive posterior|; fraction of items whose stated P is strictly closer to Bayes than to
naive; parse/refusal rate; mean output tokens. Refusals and API errors are missing data, never answers.

Decision rule (fixed in advance):
- BAYES-ROBUST: Bayes agreement on discriminating items >= 0.80 in EACH of F1, F2 and F3, AND the F4 stated P is
  closer to Bayes than to naive on >= 70% of F4 items (with a parsed P).
- HEURISTIC: naive agreement on discriminating items >= 0.80 in at least two of F1-F3.
- otherwise MIXED.
One VERDICT_BAYES line per model. Discriminating-item counts: {disc}
"""


def write_prereg(items):
    disc = {f: sum(1 for it in items if it["family"] == f and it["disc"]) for f in FAMILIES}
    txt = PREREG.format(seed=SEED, models=", ".join(f"{a}={m} [{md}]" for a, (m, md) in MODELS.items()),
                        disc=json.dumps(disc))
    (OUT / "prereg.md").write_text(txt, encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------- runner
def do_call(cache, alias, it, retries=2):
    mid, mode = MODELS[alias]
    kk = f"{alias}|{it['id']}"
    if kk in cache.d:
        return cache.d[kk]
    rec = None
    for att in range(retries + 1):
        r = rc.call(mid, it["prompt"], mode)
        ans = rc.parse_yes_no(r["text"]) if r["error"] is None else None
        verdict = None if ans is None else (int(ans == "YES") if it["polarity"] == 1 else int(ans == "NO"))
        rec = {"key": kk, "ok": verdict is not None, "alias": alias, "model": mid, "mode": mode, "id": it["id"],
               "family": it["family"], "ans": ans, "verdict": verdict, "p_compliant": parse_p(r["text"]),
               "text": r["text"], "in_tok": r["in_tok"], "out_tok": r["out_tok"], "cost": r["cost"], "stop": r["stop"],
               "error": r["error"], "attempt": att, "t": time.time()}
        cache.add(rec)
        err = r["error"] or ""
        if rec["ok"] or err.startswith("refusal") or err.startswith(("402", "403")):
            break
    return rec


def cmd_run(a):
    items = load_items()
    write_prereg(items)
    if a.limit:
        per = {}
        sel = []
        for it in items:
            if per.get(it["family"], 0) < a.limit:
                sel.append(it)
                per[it["family"]] = per.get(it["family"], 0) + 1
        items = sel
    for alias in a.models:
        cache = rc.Cache(OUT / f"calls_{alias}.jsonl")
        todo = [it for it in items if f"{alias}|{it['id']}" not in cache.d]
        print(f"[{alias}] {len(cache.d)} cached, {len(todo)} to do, workers={a.workers}, budget=${a.budget}", flush=True)
        rc.run_tasks(todo, lambda it: do_call(cache, alias, it), workers=a.workers, budget=a.budget, label=alias)
        print(f"[{alias}] DONE spend ${rc.SPEND['usd']:.3f}", flush=True)


# -------------------------------------------------------------------------------------------------------- analysis
def fmt(b):
    return "   n/a           " if b is None else f"{b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}] n={b[3]}"


def cmd_analyse(a):
    items = {it["id"]: it for it in load_items()}
    summary, lines = {}, []
    verdict_lines = []
    for alias in MODELS:
        path = OUT / f"calls_{alias}.jsonl"
        if not path.exists():
            continue
        allrecs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        cache = rc.Cache(path)
        recs = cache.d
        n_attempt = len(allrecs)
        refusals = sum(1 for r in allrecs if (r.get("error") or "").startswith("refusal"))
        api_err = sum(1 for r in allrecs if r.get("error") and not (r["error"].startswith("refusal")))
        spend = sum(r.get("cost", 0.0) for r in allrecs)
        S = {"n_items_answered": len(recs), "n_calls": n_attempt, "refusals": refusals, "api_errors": api_err,
             "parse_fail_calls": sum(1 for r in allrecs if not r.get("ok") and not r.get("error")),
             "spend_usd": round(spend, 4), "families": {}}
        lines.append(f"\n== {alias} ({MODELS[alias][0]} [{MODELS[alias][1]}]) answered {len(recs)}/{192} items, "
                     f"calls={n_attempt}, refusals={refusals}, api_errors={api_err}, parse-fail calls="
                     f"{S['parse_fail_calls']}, spend=${spend:.3f}")
        fam_stats = {}
        for f in FAMILIES:
            fr = [r for r in recs.values() if r["family"] == f]
            ids_f = [i for i in items if items[i]["family"] == f]
            if not fr:
                continue
            dv, dg, nv, cv, cg, eb, en, cl, clg, ot, pm = [], [], [], [], [], [], [], [], [], [], 0
            inc, incg, pdv, pdg = [], [], [], []       # P-vs-ANSWER inconsistency; Bayes agreement of the P-implied decision
            for r in fr:
                it = items[r["id"]]
                ot.append(r["out_tok"])
                if it["disc"]:
                    dv.append(float(r["verdict"] == it["bayes_dec"])); dg.append(r["id"])
                    nv.append(float(r["verdict"] == it["naive_dec"]))
                else:
                    cv.append(float(r["verdict"] == it["bayes_dec"])); cg.append(r["id"])
                if r["p_compliant"] is None:
                    pm += 1
                    continue
                p = r["p_compliant"] / 100
                if abs(p - 0.5) > 1e-9:
                    inc.append(float(int(p > 0.5) != r["verdict"])); incg.append(r["id"])
                    if it["disc"]:
                        pdv.append(float(int(p > 0.5) == it["bayes_dec"])); pdg.append(r["id"])
                eb.append(abs(p - it["bayes_p"])); en.append(abs(p - it["naive_p"]))
                if abs(it["bayes_p"] - it["naive_p"]) > 1e-6:
                    cl.append(float(abs(p - it["bayes_p"]) < abs(p - it["naive_p"]))); clg.append(r["id"])
            FS = {"n_answered": len(fr), "n_items": len(ids_f), "n_disc": len(dg), "n_ctrl": len(cg),
                  "bayes_agree_disc": rc.boot(dv, dg), "naive_agree_disc": rc.boot(nv, dg),
                  "control_acc": rc.boot(cv, cg),
                  "abs_err_bayes": rc.boot(eb, clg if len(clg) == len(eb) else list(range(len(eb)))),
                  "abs_err_naive": rc.boot(en, clg if len(clg) == len(en) else list(range(len(en)))),
                  "closer_to_bayes": rc.boot(cl, clg), "p_missing": pm,
                  "p_answer_inconsistent": rc.boot(inc, incg), "bayes_agree_disc_by_P": rc.boot(pdv, pdg),
                  "mean_out_tok": round(float(np.mean(ot)), 1) if ot else None}
            fam_stats[f] = FS
            S["families"][f] = FS
            lines.append(f"  {f}: answered {len(fr)}/{len(ids_f)} | Bayes-agree(disc) {fmt(FS['bayes_agree_disc'])} | "
                         f"naive-agree(disc) {fmt(FS['naive_agree_disc'])} | control acc {fmt(FS['control_acc'])}")
            lines.append(f"      |P-Bayes| {fmt(FS['abs_err_bayes'])} | |P-naive| {fmt(FS['abs_err_naive'])} | "
                         f"closer-to-Bayes {fmt(FS['closer_to_bayes'])} | P missing {pm} | out_tok {FS['mean_out_tok']}")
            lines.append(f"      P-vs-ANSWER inconsistent {fmt(FS['p_answer_inconsistent'])} | Bayes-agree(disc) by "
                         f"P-implied decision {fmt(FS['bayes_agree_disc_by_P'])}")
        # verdict
        def g(f, k):
            v = fam_stats.get(f, {}).get(k)
            return None if v is None else v[0]
        ba = [g(f, "bayes_agree_disc") for f in ("F1", "F2", "F3")]
        na = [g(f, "naive_agree_disc") for f in ("F1", "F2", "F3")]
        f4c = g("F4", "closer_to_bayes")
        if all(x is not None and x >= 0.80 for x in ba) and f4c is not None and f4c >= 0.70:
            tag = "BAYES-ROBUST"
        elif sum(1 for x in na if x is not None and x >= 0.80) >= 2:
            tag = "HEURISTIC"
        else:
            tag = "MIXED"
        if len(recs) < 192 * 0.9:
            tag += " (INCOMPLETE)"
        vl = (f"VERDICT_BAYES {alias}: {tag} | Bayes-agree disc F1/F2/F3 = "
              + "/".join("n/a" if x is None else f"{x:.2f}" for x in ba)
              + " | naive-agree disc F1/F2/F3 = " + "/".join("n/a" if x is None else f"{x:.2f}" for x in na)
              + f" | F4 closer-to-Bayes = {'n/a' if f4c is None else f'{f4c:.2f}'}"
              + (" | DRYRUN (random answers; not evidence)" if os.environ.get("DRYRUN") == "1" else ""))
        S["VERDICT_BAYES"] = vl
        verdict_lines.append(vl)
        summary[alias] = S
    # provider spend
    prov = {"anthropic": 0.0, "openrouter": 0.0}
    for alias, S in summary.items():
        prov["openrouter" if "/" in MODELS[alias][0] else "anthropic"] += S["spend_usd"]
    summary["spend_by_provider"] = {k: round(v, 3) for k, v in prov.items()}
    # compact table
    lines.append("\nCOMPACT TABLE (disc = discriminating items; agree = decision agreement; errors are mean |P/100 - posterior|)")
    lines.append(f"{'model':12s} {'fam':3s} {'Bayes-agree disc [95%CI]':26s} {'naive-agree':11s} {'ctrl acc':9s} "
                 f"{'|P-Bayes|':9s} {'|P-naive|':9s} {'closerB':7s} {'n':>3s}")
    for alias, S in summary.items():
        if alias == "spend_by_provider":
            continue
        for f, FS in S["families"].items():
            b, nn, c = FS["bayes_agree_disc"], FS["naive_agree_disc"], FS["control_acc"]
            eb, en, cl = FS["abs_err_bayes"], FS["abs_err_naive"], FS["closer_to_bayes"]
            lines.append(f"{alias:12s} {f:3s} "
                         f"{(f'{b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}]' if b else 'n/a'):26s} "
                         f"{(f'{nn[0]:.2f}' if nn else 'n/a'):11s} {(f'{c[0]:.2f}' if c else 'n/a'):9s} "
                         f"{(f'{eb[0]:.3f}' if eb else 'n/a'):9s} {(f'{en[0]:.3f}' if en else 'n/a'):9s} "
                         f"{(f'{cl[0]:.2f}' if cl else 'n/a'):7s} {FS['n_answered']:3d}")
    lines.append("")
    lines += verdict_lines
    lines.append(f"\nSPEND by provider: {summary['spend_by_provider']}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "analysis.txt").write_text(txt, encoding="utf-8")


def cmd_items(a):
    items = load_items()
    write_prereg(items)
    for f in FAMILIES:
        fi = [it for it in items if it["family"] == f]
        nd = sum(it["disc"] for it in fi)
        same_p = sum(abs(it["bayes_p"] - it["naive_p"]) < 1e-6 for it in fi)
        print(f"{f}: {len(fi)} items, {nd} discriminating, {same_p} with bayes_p==naive_p, "
              f"polarity split {sum(it['polarity'] for it in fi)}/{len(fi)}")
    print(f"items -> {OUT / 'items.json'}; prereg -> {OUT / 'prereg.md'}")
    if a.show:
        for f in FAMILIES:
            it = next(x for x in items if x["family"] == f and x["disc"] == (f != "F4"))
            print(f"\n----- {it['id']} bayes_p={it['bayes_p']:.3f} dec={it['bayes_dec']} naive_p={it['naive_p']:.3f} "
                  f"dec={it['naive_dec']} disc={it['disc']}\n{it['prompt']}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    i = sp.add_parser("items")
    i.add_argument("--show", action="store_true")
    r = sp.add_parser("run")
    r.add_argument("--models", nargs="+", required=True, choices=list(MODELS))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--budget", type=float, default=5.0)
    r.add_argument("--limit", type=int, default=0, help="items per family (smoke tests)")
    sp.add_parser("analyse")
    a = ap.parse_args()
    {"items": cmd_items, "run": cmd_run, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    main()
