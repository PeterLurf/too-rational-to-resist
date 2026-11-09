"""Consolidated analysis of every R3-B herding run for the paper (reads all cache folders, no API calls).

Per (model, mode, condition): BHW-cascade override rate, no-cascade override rate, solo signal-following, stated
probability error against (a) the exact BHW/Bayes posterior and (b) a naive posterior that counts every visible verdict
as a signal, positional accuracy (agents 3-12) and final accuracy; paired protocol gains (probs - verdicts,
signals - verdicts) on shared episodes with a cluster bootstrap over episodes.

  python tools/herd_paper.py            -> prints a report, writes r3-herding/paper_summary.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r3_common as rc  # noqa: E402
import r3_herding as h  # noqa: E402
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

BASE = pathlib.Path(SWARM_ROOT + "/r3-herding")
FOLDERS = ["", "xfamily", "probs", "pow_gpt", "pow_gemini", "pow_claude", "open_qwen14think", "open_qwen14fast",
           "open_llama8", "fast_frontier", "plant_open", "plant_claude", "plant_haiku", "plant_claude2",
           "ladder_open", "ladder_open2", "adp_a_dsflash",
           "adp_a_gemma26", "adp_a_gemma31"]
LABEL = {("claude-opus-5-5", "think"): "Opus 5.5", ("openai/gpt-6.1-sol", "think"): "GPT-6.1",
         ("google/gemini-4-argon", "think"): "Gemini 4 Argon", ("claude-sonnet-5-5", "think"): "Sonnet 5.5",
         ("claude-haiku-5-5", "think"): "Haiku 5.5 (reasoning)", ("claude-haiku-5-5", "fast"): "Haiku 5.5 (fast)",
         ("qwen/qwen3-14b", "think"): "Qwen3-14B (reasoning)", ("qwen/qwen3-14b", "fast"): "Qwen3-14B (no-think)",
         ("meta-llama/llama-3.1-8b-instruct", "fast"): "Llama-3.1-8B",
         ("claude-sonnet-5-5", "fast"): "Sonnet 5.5 (fast)", ("openai/gpt-6.1-sol", "fast"): "GPT-6.1 (minimal)",
         ("openai/gpt-6-luna", "think"): "GPT-6-Luna", ("deepseek/deepseek-v4-flash", "think"): "DeepSeek-V4-Flash (OpenRouter, partial)",
         ("openai/gpt-oss-120b", "think"): "GPT-OSS-120B", ("meta-llama/llama-3.3-70b-instruct", "fast"): "Llama-3.3-70B",
         ("adaption:deepseek-v4-flash", "default"): "DeepSeek-V4-Flash", ("adaption:gemma-4-26b-a4b-it", "default"): "Gemma-4-26B-A4B",
         ("adaption:gemma-4-31b-it", "default"): "Gemma-4-31B"}
ORDER = list(LABEL.values())


def load():
    recs = {}
    for f in FOLDERS:
        p = BASE / f / "calls.jsonl" if f else BASE / "calls.jsonl"
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok"):
                recs[r["key"]] = r
    return list(recs.values())


def naive_post(own, prev):
    return h.posterior(own + sum(1 if v == 1 else -1 for v in prev))


def theory(ns=(4, 8, 12, 24, 48, 96), n_ep=200_000, seed=7):
    """Expected accuracy of perfectly rational agents under the verdict protocol (BHW cascade) vs full pooling
    (signals/probs), by chain length N: final agent, and mean over agents 3..N. Tie at the agent's own posterior 0.5
    is broken toward the own signal, as in rational_chain."""
    rng = np.random.default_rng(seed)
    res = {}
    for n in ns:
        st = rng.integers(0, 2, n_ep)
        sig = np.where(rng.random((n_ep, n)) < h.P, st[:, None], 1 - st[:, None])
        s = 2 * sig - 1                                         # +1 PASS / -1 FAIL
        # full pooling: agent k decides on sum of signals 0..k (ties -> own signal)
        cs = np.cumsum(s, axis=1)
        pool = np.where(cs == 0, sig, (cs > 0).astype(int))
        # BHW verdicts: public lead d; informative while |d| < 2
        d = np.zeros(n_ep, int)
        ver = np.zeros((n_ep, n), int)
        for k in range(n):
            net = d + s[:, k]
            v = np.where(net == 0, sig[:, k], (net > 0).astype(int))
            ver[:, k] = v
            d = np.where(np.abs(d) < 2, d + (2 * v - 1), d)
        acc = lambda m: (m == st[:, None])  # noqa: E731
        res[n] = {"verdicts_final": round(float(acc(ver)[:, -1].mean()), 4),
                  "pooling_final": round(float(acc(pool)[:, -1].mean()), 4),
                  "verdicts_pos3": round(float(acc(ver)[:, 2:].mean()), 4) if n > 2 else None,
                  "pooling_pos3": round(float(acc(pool)[:, 2:].mean()), 4) if n > 2 else None}
    return res


def planted_theory(n=12, n_ep=200_000, seed=11):
    """Rational agents (who assume every predecessor is a rational auditor) after two sock-puppets pointing to the
    wrong state: mean accuracy of genuine agents 3..N and of the last agent, per protocol."""
    rng = np.random.default_rng(seed)
    lam = np.log(h.P / (1 - h.P))
    st = rng.integers(0, 2, n_ep)
    sig = np.where(rng.random((n_ep, n)) < h.P, st[:, None], 1 - st[:, None])
    s = 2 * sig - 1
    toward = np.where(st == 1, 1, -1)               # +1 if truth is COMPLIANT
    out = {}
    for name, offset in (("verdicts_plant", None), ("signals_plant", -2.0), ("probs_plant", -2.0),
                         ("probs_plantx", -np.log(0.99 / 0.01) / lam)):
        if offset is None:
            # public lead already at -2 against the truth: a cascade on the wrong state from agent 3 on
            ver = np.repeat((1 - st)[:, None], n - 2, axis=1)
        else:
            cs = offset * toward[:, None] + np.cumsum(s[:, 2:], axis=1)
            ver = np.where(np.isclose(cs, 0), sig[:, 2:], (cs > 0).astype(int))
        acc = ver == st[:, None]
        out[name] = {"pos_acc_genuine": round(float(acc.mean()), 4), "final_acc": round(float(acc[:, -1].mean()), 4)}
    return out


def planted(by, eps_all):
    res = {}
    for (model, mode, cond), eps in by.items():
        if "_plant" not in cond:
            continue
        lab = LABEL.get((model, mode), f"{model}|{mode}")
        base = by.get((model, mode, cond.split("_")[0]), {})
        pa, fin, allw, g, diff, gd = [], [], [], [], [], []
        for ei, e in eps.items():
            if sorted(e) != list(range(2, h.N_AGENTS)):
                continue
            st = eps_all[ei]["state"]
            v = [e[k]["verdict"] for k in range(2, h.N_AGENTS)]
            pa.append(float(np.mean([x == st for x in v])))
            fin.append(float(v[-1] == st))
            allw.append(float(all(x != st for x in v)))
            g.append(ei)
            b = base.get(ei)
            if b and sorted(b) == list(range(h.N_AGENTS)):
                diff.append(pa[-1] - float(np.mean([b[k]["verdict"] == st for k in range(2, h.N_AGENTS)])))
                gd.append(ei)
        res.setdefault(lab, {})[cond] = {"pos_acc_genuine": rc.boot(pa, g), "final_acc": rc.boot(fin, g),
                                         "all_genuine_wrong": rc.boot(allw, g),
                                         "minus_unplanted": rc.boot(diff, gd) if diff else None}
    return res


def main():
    recs = load()
    by = {}
    for r in recs:
        by.setdefault((r["model"], r["mode"], r["cond"]), {}).setdefault(r["ep"], {})[r["k"]] = r
    eps_all = {e["ep"]: e for e in h.episodes(h.MAX_EPISODES)}
    out, posacc = {}, {}
    for (model, mode, cond), eps in by.items():
        lab = LABEL.get((model, mode), f"{model}|{mode}")
        S = {"n_episodes": len(eps)}
        if cond == "solo":
            rs = [r for e in eps.values() for r in e.values()]
            S["signal_following"] = rc.boot([float(r["verdict"] == r["signal"]) for r in rs], [r["ep"] for r in rs])
            pe = [abs(r["p_compliant"] / 100 - h.posterior(1 if r["signal"] else -1)) for r in rs
                  if r["p_compliant"] is not None]
            S["p_err_bayes"] = round(float(np.mean(pe)), 5) if pe else None
        else:
            o_c, g_c, o_n, g_n, pe_b, pe_n, deep_b, deep_n, deep_closer, g_deep = [], [], [], [], [], [], [], [], [], []
            pa, fin, g_ch = {}, [], []
            ba, g_ba = [], []                                  # agreement with the exact Bayes action
            bins = {}                                          # signed public lead (own perspective) -> overrides
            for ei, e in eps.items():
                ks = sorted(e)
                if ks != list(range(h.N_AGENTS)):
                    continue                                   # complete chains only
                ver = [e[k]["verdict"] for k in ks]
                sigs = h.order(eps_all[ei], cond)
                st = eps_all[ei]["state"]
                leads = h.bhw_public(ver)
                for k in ks:
                    r = e[k]
                    own = 1 if r["signal"] == 1 else -1
                    over = float(r["verdict"] != r["signal"])
                    pub = (sum(1 if x == 1 else -1 for x in sigs[:k]) if cond in ("signals", "probs") else leads[k]) * own
                    net_abs = own + pub * own                      # Bayes net count in absolute direction
                    ba.append(float(r["verdict"] == (r["signal"] if net_abs == 0 else int(net_abs > 0))))
                    g_ba.append(ei)
                    if k > 0:
                        b = max(-3, min(3, pub))
                        bins.setdefault(b, ([], []))[0].append(over)
                        bins[b][1].append(ei)
                    if pub <= -2:
                        o_c.append(over); g_c.append(ei)
                    elif abs(pub) <= 1 and pub != -1:
                        o_n.append(over); g_n.append(ei)
                    if r["p_compliant"] is not None:
                        pb = h.posterior(own + pub * own)
                        p = r["p_compliant"] / 100
                        pe_b.append(abs(p - pb))
                        if cond in ("verdicts", "vrev"):
                            pn = naive_post(own, ver[:k])
                            pe_n.append(abs(p - pn))
                            if abs(pn - pb) >= 0.10:            # deep cascade: naive counting and BHW disagree
                                deep_b.append(abs(p - pb)); deep_n.append(abs(p - pn))
                                deep_closer.append(float(abs(p - pb) < abs(p - pn))); g_deep.append(ei)
                pa[ei] = float(np.mean([ver[k] == st for k in range(2, h.N_AGENTS)]))
                fin.append(float(ver[-1] == st)); g_ch.append(ei)
            S["override_by_lead"] = {str(b): rc.boot(*v) for b, v in sorted(bins.items())}
            S["override_cascade"] = rc.boot(o_c, g_c)
            S["bayes_action_agree"] = rc.boot(ba, g_ba)
            S["override_no_cascade"] = rc.boot(o_n, g_n)
            S["p_err_bayes"] = round(float(np.mean(pe_b)), 5) if pe_b else None
            if pe_n:
                S["p_err_naive"] = round(float(np.mean(pe_n)), 5)
            if deep_b:
                S["deep_cascade"] = {"n": len(deep_b), "p_err_bayes": round(float(np.mean(deep_b)), 5),
                                     "p_err_naive": round(float(np.mean(deep_n)), 5),
                                     "closer_to_bayes": rc.boot(deep_closer, g_deep)}
            # "privately right, publicly wrong": the 12 private signals favour the truth (strict majority) but the
            # final announced verdict is wrong
            prw = [float(fv == 0.0) for ei, fv in zip(g_ch, fin)
                   if (lambda s: s != 0 and (s > 0) == (eps_all[ei]["state"] == 1))(
                       sum(1 if x == 1 else -1 for x in eps_all[ei]["signals"]))]
            S["publicly_wrong_given_private_majority_right"] = rc.boot(prw, list(range(len(prw)))) if prw else None
            S["pos_acc_3_12"] = rc.boot(list(pa.values()), list(pa.keys()))
            S["final_acc"] = rc.boot(fin, g_ch)
            posacc[(lab, cond)] = pa
            # rational benchmark on the same episodes
            rat = {ei: float(np.mean([v == eps_all[ei]["state"] for v in h.rational_chain(h.order(eps_all[ei], cond),
                                                                                             cond)[2:]])) for ei in pa}
            S["bhw_rational_pos_acc"] = rc.boot(list(rat.values()), list(rat.keys()))
        out.setdefault(lab, {})[cond] = S
    # paired protocol gains
    for lab in list(out):
        for alt in ("probs", "signals"):
            a, b = posacc.get((lab, alt)), posacc.get((lab, "verdicts"))
            if a and b:
                common = sorted(set(a) & set(b))
                out[lab][f"gain_{alt}_minus_verdicts"] = rc.boot([a[e] - b[e] for e in common], common)
    # report
    f = lambda x: "  --  " if not x else f"{x[0]:.3f} [{x[1]:.3f},{x[2]:.3f}] n={x[3]}"  # noqa: E731
    for lab in sorted(out, key=lambda s: ORDER.index(s) if s in ORDER else 99):
        print(f"\n######## {lab}")
        for cond, S in sorted(out[lab].items()):
            if cond.startswith("gain_"):
                print(f"  {cond:28s} {f(S)}")
                continue
            print(f"  [{cond}] eps={S['n_episodes']}")
            for k, v in S.items():
                if k == "n_episodes":
                    continue
                if k == "override_by_lead":
                    print(f"     {k:24s} " + "  ".join(f"{b}:{x[0]:.2f}(n{x[3]})" for b, x in v.items() if x))
                elif isinstance(v, dict):
                    print(f"     {k:24s} n={v['n']} err_bayes={v['p_err_bayes']} err_naive={v['p_err_naive']} "
                          f"closer_to_bayes={f(v['closer_to_bayes'])}")
                else:
                    print(f"     {k:24s} {f(v) if isinstance(v, list) or v is None else v}")
    out["_planted"] = planted(by, eps_all)
    out["_planted_theory"] = planted_theory()
    print("\nPLANTED (two sock-puppets first; genuine agents 3-12):")
    for lab, d in out["_planted"].items():
        for cond, S in sorted(d.items()):
            print(f"  {lab:24s} {cond:15s} pos_acc {f(S['pos_acc_genuine'])} | final {f(S['final_acc'])} | "
                  f"all-wrong {f(S['all_genuine_wrong'])} | minus unplanted {f(S['minus_unplanted'])}")
    print("PLANTED THEORY (rational): " + json.dumps(out["_planted_theory"]))
    out["_theory"] = theory()
    print("\nTHEORY (rational agents, p=0.65, 200k episodes): " + json.dumps(out["_theory"]))
    (BASE / "paper_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n-> {BASE / 'paper_summary.json'}  ({len(recs)} records)")


if __name__ == "__main__":
    main()
