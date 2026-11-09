"""LaTeX table for the plant sweep (number and position of fake verdicts), generated from
r3-herding/sweep_summary.json (tools/sweep_analyse.py). Cells: genuine accuracy (rational benchmark on the same
signals and fake positions). Regenerate:  uv run --isolated --no-project python figs/make_tab_sweep.py
"""
import json
import os
import pathlib

import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))
S = json.load(open(pathlib.Path(SWARM_ROOT) / "r3-herding/sweep_summary.json",
                   encoding="utf-8"))
ROWS = [("verdicts_plant1", "one, position 1"), ("verdicts_plant", "two, positions 1--2"),
        ("verdicts_plant3", "three, positions 1--3"), ("verdicts_plantmid", "two, positions 5--6")]
COLS = [("claude-sonnet-5-5|think", "Sonnet"), ("claude-haiku-5-5|think", "Haiku (reasoning)"),
        ("claude-haiku-5-5|fast", "Haiku (fast)")]


def cell(key):
    v = S.get(key)
    if not v or v["acc"][3] < 10:
        return "--"
    return f"{v['acc'][0]:.2f} ({v['rational_acc']:.2f}), $n={v['acc'][3]}$"


lines = ["\\begin{tabular}{l" + "c" * len(COLS) + "}", "\\toprule",
         "fake verdicts & " + " & ".join(c[1] for c in COLS) + " \\\\", "\\midrule"]
for cond, lab in ROWS:
    lines.append(lab + " & " + " & ".join(cell(f"{c}|{cond}") for c, _ in COLS) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}"]
pathlib.Path("figs/tab_sweep.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
