"""How often each model names the cascade structure in its visible answer (verdicts protocol), and how often GPT-6.1
flags the extreme planted probabilities as impossible. Visible text only (Claude thinking is not returned).

  uv run --isolated --no-project python tools/cascade_mentions.py
"""
import collections
import json
import pathlib
import re
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

B = pathlib.Path(SWARM_ROOT + "/r3-herding")
FOLDERS = ["", "xfamily", "pow_gpt", "pow_gemini", "open_qwen14think", "open_qwen14fast", "open_llama8", "fast_frontier"]
CASC = re.compile(r"cascade|herd|bikhchandani", re.I)
IMPOSSIBLE = re.compile(r"(cannot|can't|could not|couldn't) (?:themselves )?(be|follow|arise|come)|inconsistent|impossible|"
                        r"not (?:be )?(?:exact|calibrated|valid|consistent|possible)|(?:could|can|should) only "
                        r"(?:be|report|obtain|yield|justify)|should (?:have been|report|be) either|incompatible|"
                        r"too (?:high|low|extreme) to be", re.I)


def records(folder):
    p = B / folder / "calls.jsonl" if folder else B / "calls.jsonl"
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("ok"):
            yield r


def main():
    seen, n, c = set(), collections.Counter(), collections.Counter()
    for f in FOLDERS:
        for r in records(f):
            if r["cond"] != "verdicts" or r["key"] in seen:
                continue
            seen.add(r["key"])
            k = (r["model"], r["mode"])
            n[k] += 1
            c[k] += bool(CASC.search(r["text"] or ""))
    print("names cascade/herding in visible answer (verdicts protocol):")
    for k in sorted(n):
        print(f"  {k[0]:34s} {k[1]:6s} {c[k] / n[k]:.3f}  (n={n[k]})")
    po = list(records("plant_open"))
    for cond in ("probs_plantx", "probs_plant"):
        rs = [r for r in po if r["model"] == "openai/gpt-6.1-sol" and r["cond"] == cond]
        k2 = [r for r in rs if r["k"] == 2]
        face = sum(min(r["p_compliant"], 100 - r["p_compliant"]) <= 2.0 for r in k2 if r["p_compliant"] is not None)
        print(f"GPT-6.1 {cond}: flags planted reports as impossible at k=2 {sum(bool(IMPOSSIBLE.search(r['text'] or '')) for r in k2)}"
              f"/{len(k2)}, all genuine positions {sum(bool(IMPOSSIBLE.search(r['text'] or '')) for r in rs)}/{len(rs)}; "
              f"k=2 stated P within 2 points of 0/100: {face}/{len(k2)}")


if __name__ == "__main__":
    main()
