"""LaTeX summary table for the main text: one column per model configuration, three rows that carry the paper's
argument. Generated from twin-swap/summary_{fast,reason}.json (tools/twin_swap.py) and
r3-herding/paper_summary.json (tools/herd_paper.py). Regenerate:  python figs/make_tab_summary.py

  capture      share of code-review pairs (judged right alone) whose verdict the other twin's transcript flips,
               in the worse of the two directions, in the mode that matches the herding run
  Bayes        agreement with the exact Bayes action on honest verdict chains
  hijacked     share of episodes in which two planted verdicts placed first turn every genuine agent wrong
"""
import json
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))
ROOT = _pl.Path(SWARM_ROOT)

CAP = {m: json.load(open(ROOT / f"twin-swap/summary_{m}.json", encoding="utf-8"))["rows"] for m in ("fast", "reason")}
HERD = json.load(open(ROOT / "r3-herding/paper_summary.json", encoding="utf-8"))
OVERSEER_SRC = "anthropic-api/claude-sonnet-5-5@r1"

# (column label, capture model id, capture mode, herding key); capture mode matches the herding run's mode
COLS = [("Opus", "claude-opus-5-5", "reason", "Opus 5.5"),
        ("GPT-6.1", "openai/gpt-6.1-sol", "reason", "GPT-6.1"),
        ("Gemma\\\\4-31B", "google/gemma-4-31b-it", "reason", "Gemma-4-31B"),
        ("Sonnet", "claude-sonnet-5-5", "reason", "Sonnet 5.5"),
        ("Haiku\\\\reasoning", "claude-haiku-5-5", "reason", "Haiku 5.5 (reasoning)"),
        ("Haiku\\\\fast", "claude-haiku-5-5", "fast", "Haiku 5.5 (fast)"),
        ("GPT-OSS\\\\120B", "openai/gpt-oss-120b", "reason", "GPT-OSS-120B"),
        ("Qwen3-14B\\\\fast", "qwen/qwen3-14b", "fast", "Qwen3-14B (no-think)"),
        ("Llama-3.3\\\\70B", "meta-llama/llama-3.3-70b-instruct", "fast", "Llama-3.3-70B")]


def capture(model, mode):
    r = CAP[mode].get(f"{model}|{OVERSEER_SRC}")
    vals = [r[k][0] for k in ("sus_clear", "sus_alarm") if r and r.get(k)]
    return max(vals) if vals else None


def bayes(key):
    v = HERD[key]["verdicts"].get("bayes_action_agree")
    return v[0] if v else None


def hijack(key):
    v = HERD["_planted"].get(key, {}).get("verdicts_plant", {}).get("all_genuine_wrong")
    return v[0] if v else None


def fmt(x):
    return "--" if x is None else f"{x:.2f}"


def head(label):
    return "\\begin{tabular}[b]{@{}c@{}}" + label + "\\end{tabular}"


rows = [("captured by a false review", [capture(m, md) for _, m, md, _ in COLS]),
        ("takes the Bayes action", [bayes(k) for *_, k in COLS]),
        ("hijacked by two fakes", [hijack(k) for *_, k in COLS])]
lines = ["\\begin{tabular}{@{}l" + "c" * len(COLS) + "@{}}", "\\toprule",
         " & " + " & ".join(head(c[0]) for c in COLS) + " \\\\", "\\midrule"]
for lab, vals in rows:
    lines.append(lab + " & " + " & ".join(fmt(v) for v in vals) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
_pl.Path(__file__).resolve().parent.joinpath("tab_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
