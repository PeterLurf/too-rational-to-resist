"""Plant sweep (number and position of fake verdicts): genuine accuracy and full capture vs rational (BHW) agents on
the same signals and fake positions. Rational agents treat every visible verdict as a rational predecessor's
(informative while the public lead |d| < 2), ties toward the own signal.
  uv run --isolated --no-project --with numpy python tools/sweep_analyse.py
"""
import json, pathlib, sys, collections
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import r3_herding as h  # noqa: E402
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))
B = pathlib.Path(SWARM_ROOT + "/r3-herding")
CONDS = {"verdicts_plant1": [0], "verdicts_plant": [0, 1], "verdicts_plant3": [0, 1, 2], "verdicts_plantmid": [4, 5]}
FOLDERS = ["sweep_sonnet", "sweep_haiku", "plant_claude", "plant_haiku", "plant_open", "plant_claude2"]


def rational(ep, fakes):
    st, ver, acc = ep["state"], [], []
    for k in range(h.N_AGENTS):
        if k in fakes:
            ver.append(1 - st); continue
        s = ep["signals"][k]; own = 1 if s == 1 else -1
        net = own + h.bhw_public(ver)[k]
        a = s if net == 0 else int(net > 0)
        ver.append(a); acc.append(a == st)
    return acc


def boot(x, nb=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return [round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(x)]


def main():
    recs = {}
    for f in FOLDERS:
        p = B / f / "calls.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                try: r = json.loads(line)
                except json.JSONDecodeError: continue
                if r.get("ok"): recs[r["key"]] = r
    E = {e["ep"]: e for e in h.episodes(h.MAX_EPISODES)}
    out = {}
    for mm in [("claude-sonnet-5-5", "think"), ("claude-haiku-5-5", "think"), ("claude-haiku-5-5", "fast")]:
        for cond, fakes in CONDS.items():
            by = collections.defaultdict(dict)
            for r in recs.values():
                if (r["model"], r["mode"]) == mm and r["cond"] == cond: by[r["ep"]][r["k"]] = r
            gen = [k for k in range(h.N_AGENTS) if k not in fakes]
            full = {e: d for e, d in by.items() if sorted(d) == gen and e < 30}
            if not full: continue
            acc = [float(np.mean([d[k]["verdict"] == E[e]["state"] for k in gen])) for e, d in full.items()]
            allw = [float(all(d[k]["verdict"] != E[e]["state"] for k in gen)) for e, d in full.items()]
            racc = [float(np.mean(rational(E[e], fakes))) for e in full]
            rall = [float(not any(rational(E[e], fakes))) for e in full]
            key = f"{mm[0]}|{mm[1]}|{cond}"
            out[key] = {"acc": boot(acc), "all_wrong": boot(allw), "rational_acc": round(float(np.mean(racc)), 3),
                        "rational_all_wrong": round(float(np.mean(rall)), 3)}
            o = out[key]
            print(f"{mm[0]:18s} {mm[1]:5s} {cond:18s} n={len(full):2d}  acc {o['acc'][0]:.2f} [{o['acc'][1]:.2f},{o['acc'][2]:.2f}]"
                  f" (rational {o['rational_acc']:.2f})  all-wrong {o['all_wrong'][0]:.2f} (rational {o['rational_all_wrong']:.2f})")
    (B / "sweep_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
