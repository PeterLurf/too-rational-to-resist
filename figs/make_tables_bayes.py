"""LaTeX tables for bayesian_swarms.tex, generated from r3-herding/paper_summary.json (tools/herd_paper.py) so that no
number is transcribed by hand. Writes figs/tab_herding.tex and figs/tab_planted.tex.
Regenerate:  uv run --isolated --no-project python figs/make_tables_bayes.py
"""
import json
import os
import pathlib
import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))

ART = pathlib.Path(SWARM_ROOT)
H = json.load(open(ART / "r3-herding/paper_summary.json", encoding="utf-8"))
ORDER = ["Opus 5.5", "GPT-6.1", "Gemini 4 Argon", "Sonnet 5.5", "Sonnet 5.5 (fast)", "GPT-6.1 (minimal)",
         "Haiku 5.5 (reasoning)", "Haiku 5.5 (fast)", "Qwen3-14B (reasoning)", "Qwen3-14B (no-think)", "Llama-3.1-8B"]


def m(v, ci=False, nd=2):
    if not v:
        return "--"
    s = f"{v[0]:.{nd}f}"
    return s + (f" [{v[1]:.{nd}f}, {v[2]:.{nd}f}]" if ci else "")


def herding():
    rows = []
    for lab in ORDER:
        d = H.get(lab, {})
        v = d.get("verdicts")
        if not v:
            continue
        so = d.get("solo", {}).get("signal_following")
        dc = v.get("deep_cascade") or {}
        pr = d.get("probs", {})
        rows.append(" & ".join([
            lab, str(v["pos_acc_3_12"][3] if v.get("pos_acc_3_12") else 0), m(v.get("override_cascade")), m(v.get("override_no_cascade")), m(so),
            f"{v['p_err_bayes']:.3f}" if v.get("p_err_bayes") is not None else "--",
            m(dc.get("closer_to_bayes")) if dc else "--",
            m(v.get("pos_acc_3_12")), m(pr.get("pos_acc_3_12")), m(v.get("bhw_rational_pos_acc")),
            m(v.get("publicly_wrong_given_private_majority_right")),
            m(pr.get("publicly_wrong_given_private_majority_right"))]) + r" \\")
    head = (r"\begin{tabular}{lrcccccccccc}" "\n" r"\toprule" "\n"
            r" & chains & \multicolumn{2}{c}{override} & solo & $|P-P_B|$ & deep & \multicolumn{3}{c}{pos.\ acc.\ 3--12}"
            r" & \multicolumn{2}{c}{publicly wrong} \\" "\n"
            r"\cmidrule(lr){3-4}\cmidrule(lr){8-10}\cmidrule(lr){11-12}" "\n"
            r"model & & casc. & none & follow & verd. & $\to$B & verd. & prob. & Bayes & verd. & prob. \\" "\n"
            r"\midrule")
    (pathlib.Path("figs") / "tab_herding.tex").write_text(
        head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n", encoding="utf-8")


def planted():
    P = H.get("_planted", {})
    T = H.get("_planted_theory", {})
    conds = [("verdicts_plant", "verdicts"), ("signals_plant", "spot checks"), ("probs_plant", "probs (calibrated)"),
             ("probs_plantx", "probs (extreme)")]
    rows = [" & ".join(["rational benchmark"] + [f"{T[c]['pos_acc_genuine']:.2f} / {T[c]['final_acc']:.2f}"
                                                  for c, _ in conds]) + r" \\", r"\midrule"]
    for lab in ORDER:
        if lab not in P:
            continue
        cells = []
        for c, _ in conds:
            s = P[lab].get(c)
            cells.append("--" if not s or not s["pos_acc_genuine"] else
                         f"{s['pos_acc_genuine'][0]:.2f} / {s['final_acc'][0]:.2f}")
        rows.append(" & ".join([lab] + cells) + r" \\")
    head = (r"\begin{tabular}{lcccc}" "\n" r"\toprule" "\n"
            "shared after two sock-puppets & " + " & ".join(n for _, n in conds) + r" \\" "\n" r"\midrule")
    (pathlib.Path("figs") / "tab_planted.tex").write_text(
        head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n", encoding="utf-8")


def stress():
    """Bayesian stress vignettes (tools/r3_bayes.py, r3-bayes/summary.json)."""
    S = json.load(open(ART / "r3-bayes/summary.json", encoding="utf-8"))
    names = [("opus", "Opus 5.5"), ("gpt", "GPT-6.1"), ("gemini", "Gemini 4 Argon"), ("sonnet", "Sonnet 5.5"),
             ("haikuthink", "Haiku 5.5 (reasoning)"), ("haikufast", "Haiku 5.5 (fast)"),
             ("qwen14think", "Qwen3-14B (reasoning)"), ("qwen14fast", "Qwen3-14B (no-think)"),
             ("llama8", "Llama-3.1-8B")]
    rows = []
    for key, lab in names:
        F = S[key]["families"]
        cells = [m(F[f]["bayes_agree_disc"]) for f in ("F1", "F2", "F3")]
        cells.append(f"{max(F[f]['abs_err_bayes'][0] for f in ('F1', 'F2', 'F3')):.3f}")
        cells.append(m(F["F4"]["closer_to_bayes"]))
        cells.append(f"{F['F4']['abs_err_bayes'][0]:.3f} / {F['F4']['abs_err_naive'][0]:.3f}")
        rows.append(" & ".join([lab] + cells) + r" \\")
    head = (r"\setlength{\tabcolsep}{4pt}" "\n" r"\begin{tabular}{lcccccc}" "\n" r"\toprule" "\n"
            r" & \multicolumn{3}{c}{decision $=$ Bayes $\ne$ count} & max $|P-P_B|$ & \multicolumn{2}{c}{F4 cascade}"
            r" \\" "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}" "\n"
            r"model & F1 prior & F2 prec. & F3 echo & F1--F3 & closer & err B / count \\"
            "\n" r"\midrule")
    (pathlib.Path("figs") / "tab_stress.tex").write_text(
        head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n", encoding="utf-8")


if __name__ == "__main__":
    herding()
    planted()
    stress()
    print("wrote figs/tab_herding.tex, figs/tab_planted.tex, figs/tab_stress.tex")
