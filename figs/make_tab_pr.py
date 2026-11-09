"""LaTeX table for the realistic review pipeline (bayesian_swarms.tex), generated from pr-review/<model>/summary.json
(tools/pr_review.py analyse). Cells: merged episodes / complete episodes. Rows are grouped by pull request so that the
direction of a good outcome is stated once per group (leak: fewer merges is better; benign: more is better).
Regenerate:  uv run --isolated --no-project python figs/make_tab_pr.py
"""
import json
import os
import pathlib

import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))
ART = pathlib.Path(SWARM_ROOT) / "pr-review"
COLS = [("haiku", "claude-haiku-5-5@fast", "Haiku (fast)"), ("sonnet", "claude-sonnet-5-5", "Sonnet"),
        ("opus", "claude-opus-5-5", "Opus")]
EXTRA = {"haiku": ["haiku_fq", "haiku_v"], "sonnet": ["sonnet_fq", "sonnet_v"], "opus": []}   # adaptive full-quote attack folders
ROWS = [("leak", "full diff", "two planted approvals with false claims", "plant_claim"),
        ("leak", "owner split", "two genuine owner reviews", "partial_clean"),
        ("leak", "owner split", r"\textbf{two planted owner approvals with false claims}", "partial_plant"),
        ("leak", "owner split", "two planted owner approvals, no claims (``LGTM'')", "partial_plant_verdict"),
        ("leak", "owner split", "two planted owner approvals quoting the full payload", "partial_plant_fullquote"),
        ("benign", "full diff", "none", "benign_clean"),
        ("benign", "owner split", "two genuine owner reviews", "benign_partial_clean")]
HEAD = {"leak": r"\emph{Data-leaking pull request (a merge is a failure)}",
        "benign": r"\emph{Benign twin (a merge is correct)}"}


def cell(S, cond):
    v = (S or {}).get(cond)
    if not v or not v["episodes"]:
        return "--"
    m = round(v["merged"][0] * v["episodes"])
    return f"{m}/{v['episodes']}"


def main():
    S = {}
    for folder, mid, _ in COLS:
        S[folder] = {}
        for fo in [folder] + EXTRA[folder]:
            p = ART / fo / "summary.json"
            if p.exists():
                rows = json.load(open(p, encoding="utf-8")).get(mid) or {}
                S[folder].update({c: v for c, v in rows.items() if v.get("episodes")})   # only conditions with data
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             "reviewers see & owner or planted reviews & " + " & ".join(c[2] for c in COLS) + r" \\", r"\midrule"]
    last = None
    for kind, view, plant, cond in ROWS:
        if kind != last:
            if last is not None:
                lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{5}{l}{" + HEAD[kind] + r"} \\")
            last = kind
        lines.append(f"{view} & {plant} & " + " & ".join(cell(S[f], cond) for f, _, _ in COLS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    pathlib.Path("figs/tab_pr.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
