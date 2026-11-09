"""Figures for the paper "Rational herds" (bayesian_swarms.tex). Every value is read from the analysis outputs of the
API experiments in the artifacts directory:
  twin-swap/summary_{fast,reason}.json      (tools/twin_swap.py analyse, branch orx/twin-transcript-swap-i4-...)
  r3-herding/paper_summary.json             (tools/herd_paper.py, branch orx/rational-herding-r3-b-...)
Regenerate:  uv run --isolated --no-project --with matplotlib --with numpy --with scipy python figs/make_figs_bayes.py
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "figs")
from orx_figstyle import BASELINE, MUTED, PALETTE, TEXT, figure_grid, panel_labels, save, use_style  # noqa: E402
import paper_theme as th  # noqa: E402

import os as _os, pathlib as _pl
SWARM_ROOT = _os.environ.get("SWARM_ARTIFACTS", str(_pl.Path(__file__).resolve().parents[1] / "data"))
ART = pathlib.Path(SWARM_ROOT)
SRC = "anthropic-api/claude-sonnet-5-5@r1"

# model display order: frontier -> mid -> small open
TIER = {"Opus 5.5": "frontier", "GPT-6.1": "frontier", "Gemini 4 Argon": "frontier", "Sonnet 5.5": "frontier",
        "Haiku 5.5 (reasoning)": "mid", "Haiku 5.5 (fast)": "mid",
        "Qwen3-14B (reasoning)": "open", "Qwen3-14B (no-think)": "open", "Llama-3.1-8B": "open"}
COL = {"Opus 5.5": "#1F3F66", "GPT-6.1": "#3F6EA8", "Gemini 4 Argon": "#86A8D2", "Sonnet 5.5": "#5E8BC4",
       "Haiku 5.5 (reasoning)": "#C98E1F", "Haiku 5.5 (fast)": "#E3BF6E",
       "Qwen3-14B (reasoning)": "#C8553D", "Qwen3-14B (no-think)": "#E39A86", "Llama-3.1-8B": "#9A958B"}


def err(v):
    return [[v[0] - v[1]], [v[2] - v[0]]]


# ------------------------------------------------------------------------------------------------ Fig: capture
# overseer ladder for the twin swap: (label, OpenRouter/Anthropic id for fast, id for reasoning, group)
LADDER = [("Opus 5.5", "claude-opus-5-5", "claude-opus-5-5", "Claude"),
          ("Sonnet 5.5", "claude-sonnet-5-5", "claude-sonnet-5-5", "Claude"),
          ("Haiku 5.5", "claude-haiku-5-5", "claude-haiku-5-5", "Claude"),
          ("Gemini 3.8 Flash", "google/gemini-3.8-flash", "google/gemini-3.8-flash", "other closed"),
          ("GPT-6-Luna", "openai/gpt-6-luna", "openai/gpt-6-luna", "other closed"),
          ("GPT-6.1", "openai/gpt-6.1-sol", "openai/gpt-6.1-sol", "other closed"),
          ("Gemini 4 Argon", "google/gemini-4-argon", "google/gemini-4-argon", "other closed"),
          ("DeepSeek-V4-Flash", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash", "open, large"),
          ("Qwen3-235B-A22B", "qwen/qwen3-235b-a22b-2507", "qwen/qwen3-235b-a22b-thinking-2507", "open, large"),
          ("Llama-4-Maverick (400B)", "meta-llama/llama-4-maverick", "meta-llama/llama-4-maverick", "open, large"),
          ("GPT-OSS-120B", "openai/gpt-oss-120b", "openai/gpt-oss-120b", "open, large"),
          ("Llama-3.3-70B", "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct", "open, 20-70B"),
          ("Llama-3.1-70B", "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-70b-instruct", "open, 20-70B"),
          ("Qwen3-32B", "qwen/qwen3-32b", "qwen/qwen3-32b", "open, 20-70B"),
          ("Gemma-4-31B", None, "google/gemma-4-31b-it", "open, 20-70B"),
          ("Gemma-3-27B", "google/gemma-3-27b-it", None, "open, 20-70B"),
          ("Mistral-Small-24B", "mistralai/mistral-small-3.2-24b-instruct", None, "open, 20-70B"),
          ("GPT-OSS-20B", "openai/gpt-oss-20b", "openai/gpt-oss-20b", "open, 20-70B"),
          ("Qwen3-14B", "qwen/qwen3-14b", "qwen/qwen3-14b", "open, small"),
          ("Llama-3.1-8B", "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-8b-instruct", "open, small")]
MIN_N = 5        # points with fewer pairs judged right alone are not plotted


GROUP_LABEL = {"Claude": "Claude", "other closed": "other closed", "open, large": "open, large",
               "open, 20-70B": "open, 20-70B", "open, small": "open, \u226414B"}


FAMILY_COL = {"Claude": th.OVER, "other closed": "#2F2E2B", "open, large": th.LEAK, "open, 20-70B": th.LEAK,
              "open, small": th.LEAK}


def fig_capture():
    """Dot plot, models sorted by mean capture: filled = answer only, open = reasoning, joined by a thin line.
    Colour = model family. CIs are omitted for legibility (median half-width 0.08, reported in the caption)."""
    S = {m: json.load(open(ART / f"twin-swap/summary_{m}.json", encoding="utf-8"))["rows"] for m in ("fast", "reason")}
    th.use_theme()
    vals = {}
    for name, fid, rid, grp in LADDER:
        for key in ("sus_clear", "sus_alarm"):
            for mode, mid in (("fast", fid), ("reason", rid)):
                r = S[mode].get(f"{mid}|{SRC}") if mid else None
                v = r[key] if r else None
                if v and v[3] >= MIN_N:
                    vals[(name, key, mode)] = v[0]
    def score(row):
        xs = [x for (n, _, _), x in vals.items() if n == row[0]]
        return np.mean(xs) if xs else 0.0
    rows = sorted(LADDER, key=score)              # least captured at the top
    fig, (axa, axb) = figure_grid(1, 2, width=th.PAPER_W, ratio=0.40, sharey=True)
    y = np.arange(len(rows))[::-1].astype(float)
    for ax, key in ((axa, "sus_clear"), (axb, "sus_alarm")):
        ax.axvspan(-0.04, 0.05, color=th.BAND, lw=0, zorder=0)
        for yi, (name, fid, rid, grp) in zip(y, rows):
            c = FAMILY_COL[grp]
            f, r = vals.get((name, key, "fast")), vals.get((name, key, "reason"))
            if f is not None and r is not None:
                ax.plot([f, r], [yi, yi], color=c, lw=1.0, alpha=0.4, zorder=2, solid_capstyle="round")
            # if both modes land on (almost) the same value, nudge them apart vertically so neither hides the other
            dy = 0.2 if (f is not None and r is not None and abs(f - r) < 0.02) else 0.0
            if f is not None:
                ax.plot(f, yi + dy, marker="o", ms=4.6, color=c, ls="none", zorder=4)
            if r is not None:
                ax.plot(r, yi - dy, marker="o", ms=4.6, mec=c, mfc="white", mew=1.1, ls="none", zorder=5)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(y[-1] - 0.7, y[0] + 0.7)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(axis="x", color=th.GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
    axa.set_yticks(y, [r[0] for r in rows])
    for lab, (name, *_rest) in zip(axa.get_yticklabels(), rows):
        lab.set_color(FAMILY_COL[dict((r[0], r[3]) for r in LADDER)[name]])
    axa.set_xlabel("talked out of a real leak (share of pairs)")
    axa.set_ylabel("overseer (least to most captured)")
    axb.set_xlabel("talked into a false alarm (share of pairs)")
    # one-row key above both panels (marker style = mode, colour = model family)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ms=4.6, color=th.INK, ls="none", label="answer only"),
               Line2D([], [], marker="o", ms=4.6, mec=th.INK, mfc="white", mew=1.1, ls="none", label="with reasoning"),
               Line2D([], [], marker="s", ms=4.6, color=th.OVER, ls="none", label="Claude"),
               Line2D([], [], marker="s", ms=4.6, color="#2F2E2B", ls="none", label="other closed"),
               Line2D([], [], marker="s", ms=4.6, color=th.LEAK, ls="none", label="open-weight")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5, fontsize=6.5, frameon=False,
               handletextpad=0.3, columnspacing=1.6)
    axb.text(0.06, y[0], "no observed flips", va="center", fontsize=6, color=th.SUB, style="italic")
    th.panel_labels([axa, axb])
    save(fig, "figs/fig_capture")


# ------------------------------------------------------------------------------------------------ Fig: herding
def fig_herding():
    H = json.load(open(ART / "r3-herding/paper_summary.json", encoding="utf-8"))
    models = [m for m in TIER if m in H and "verdicts" in H[m]]
    th.use_theme()
    fig, (axa, axb) = figure_grid(1, 2, width=th.PAPER_W, ratio=0.35, gridspec_kw={"width_ratios": [1.0, 1.15]})
    xs = [-2, -1, 0, 1, 2]
    axa.step([-2.5, -1.5, -1.5, 2.5], [1, 1, 0, 0], where="post", color="black", lw=1.0, ls="--",
             label="exact Bayes (BHW)", zorder=5)
    for i, m in enumerate(models):
        ob = H[m]["verdicts"]["override_by_lead"]
        pts = [(x, ob[str(x)]) for x in xs if str(x) in ob and ob[str(x)] and ob[str(x)][3] >= 10]
        off = (i - len(models) / 2) * 0.035
        axa.errorbar([x + off for x, _ in pts], [v[0] for _, v in pts],
                     yerr=np.array([[v[0] - v[1] for _, v in pts], [v[2] - v[0] for _, v in pts]]),
                     color=COL[m], marker="o", ms=2.6, lw=0.9, elinewidth=0.5, capsize=0, label=m)
    axa.set_xticks(xs, ["$-2$", "$-1$", "0", "+1", "+2"])
    axa.set_xlabel("informative public lead for own signal (verdicts)")
    axa.set_ylabel("P(announce against own signal)")
    axa.set_ylim(-0.03, 1.03)
    # (b) deep-cascade probability error: Bayes vs naive counting
    yy = np.arange(len(models))[::-1]
    for yi, m in zip(yy, models):
        d = H[m]["verdicts"].get("deep_cascade")
        if not d:
            continue
        axb.plot([d["p_err_bayes"], d["p_err_naive"]], [yi, yi], color=MUTED, lw=1.0, zorder=1)
        axb.scatter(d["p_err_bayes"], yi, s=14, color=COL[m], zorder=3, edgecolor=COL[m])
        axb.scatter(d["p_err_naive"], yi, s=14, facecolor="white", edgecolor=COL[m], zorder=3, linewidth=0.9)
        axb.text(1.02, yi, f"n={d['n']}", va="center", ha="left", fontsize=th.NOTE, color=BASELINE,
                 transform=axb.get_yaxis_transform())
    axb.set_yticks(yy, models)
    axb.set_xscale("log")
    axb.set_xlim(8e-4, 0.6)
    axb.set_xlabel(r"mean $|P_{\mathrm{stated}}-P_{\mathrm{ref}}|$ in deep cascades")
    # direct labels on the top row instead of a legend (Rougier et al. 2014, rule 8)
    d0 = H[models[0]]["verdicts"]["deep_cascade"]
    axb.annotate("error vs exact Bayes", (d0["p_err_bayes"], yy[0]), textcoords="offset points", xytext=(-3, 6),
                 ha="left", fontsize=th.NOTE, color=th.SUB)
    axb.annotate("vs naive count", (d0["p_err_naive"], yy[0]), textcoords="offset points", xytext=(0, 6),
                 ha="center", fontsize=th.NOTE, color=th.SUB)
    axb.set_ylim(yy[-1] - 0.6, yy[0] + 1.1)
    h, l = axa.get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, fontsize=th.NOTE, frameon=False)
    th.panel_labels([axa, axb])
    save(fig, "figs/fig_herding")


# ------------------------------------------------------------------------------------------------ Fig: exploit
EXPLOIT = ["Opus 5.5", "GPT-6.1", "Sonnet 5.5", "Haiku 5.5 (reasoning)", "DeepSeek-V4-Flash", "Gemma-4-31B",
           "Gemma-4-26B-A4B", "GPT-OSS-120B", "Haiku 5.5 (fast)", "Llama-3.3-70B", "Qwen3-14B (no-think)"]


NAME = {"Opus 5.5": "Opus", "GPT-6.1": "GPT-6.1", "Sonnet 5.5": "Sonnet", "Haiku 5.5 (reasoning)": "Haiku (reasoning)",
        "Haiku 5.5 (fast)": "Haiku (fast)", "Gemma-4-26B-A4B": "Gemma-4-26B-A4B", "GPT-OSS-120B": "GPT-OSS-120B",
        "Llama-3.3-70B": "Llama-3.3-70B", "Qwen3-14B (no-think)": "Qwen3-14B (no thinking)", "Gemma-4-31B": "Gemma-4-31B"}
LBL = {"Sonnet 5.5": ((-6, -2), "right"), "Haiku 5.5 (reasoning)": ((14, 6), "left"),
       "Haiku 5.5 (fast)": ((-22, 12), "right"), "Gemma-4-26B-A4B": ((6, 2), "left"), "GPT-OSS-120B": ((14, -1), "left"),
       "Llama-3.3-70B": ((-6, 3), "right"), "Qwen3-14B (no-think)": ((6, -7), "left")}
LEADER = ("Haiku 5.5 (reasoning)", "Haiku 5.5 (fast)", "GPT-OSS-120B")   # crowded: draw a thin leader to the dot
TOP = ("Opus 5.5", "GPT-6.1", "Gemma-4-31B")      # coincide at (~1, 1): one combined label


def _rho_txt():
    """Spearman rho with bootstrap CI, identical to tools/rho_ladder.py (models with >=10 planted, >=100 honest)."""
    from scipy.stats import spearmanr
    S = json.load(open(ART / "r3-herding/paper_summary.json", encoding="utf-8"))
    rows = []
    for m, d in S.items():
        if m.startswith("_") or "|" in m:
            continue
        x = (d.get("verdicts") or {}).get("bayes_action_agree")
        y = (S["_planted"].get(m, {}).get("verdicts_plant") or {}).get("all_genuine_wrong")
        if x and y and y[3] >= 10 and x[3] >= 100:
            rows.append((x[0], y[0]))
    x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    rho = spearmanr(x, y)[0]
    rng = np.random.default_rng(0); bs = []
    for _ in range(5000):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            bs.append(spearmanr(x[i], y[i])[0])
    return f"{rho:.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]"


def fig_exploit():
    RHO_TXT = _rho_txt()
    """(a) Agreement with the Bayes action on honest chains vs share of episodes hijacked by two planted verdicts.
    Neutral points (colour is reserved for fake verdicts / fake checks in panel b); Spearman rho from tools/rho_ladder.py.
    (b) Per model: genuine accuracy after two planted verdicts vs two planted spot checks, CIs offset per condition."""
    H = json.load(open(ART / "r3-herding/paper_summary.json", encoding="utf-8"))
    P, T = H["_planted"], H["_planted_theory"]
    th.use_theme()
    fig, (axa, axb) = figure_grid(1, 2, width=th.PAPER_W, ratio=0.35, gridspec_kw={"width_ratios": [1.1, 1.0]})
    xs_, ys_ = [], []
    for m in EXPLOIT:
        x = (H.get(m, {}).get("verdicts") or {}).get("bayes_action_agree")
        y = (P.get(m, {}).get("verdicts_plant") or {}).get("all_genuine_wrong")
        if not x or not y or y[3] < 10:
            continue
        xs_.append(x[0]); ys_.append(y[0])
        # family colour (red is reserved for fake verdicts in panel b, so open-weight models are grey here)
        c = th.OVER if m.split(" ")[0] in ("Opus", "Sonnet", "Haiku") else ("#2F2E2B" if m == "GPT-6.1" else "#8C8A84")
        th.soft_errorbar(axa, x[0], y[0], (y[1], y[2]), th.STONE, axis="y", lw=1.0, alpha=0.35)
        th.soft_errorbar(axa, x[0], y[0], (x[1], x[2]), th.STONE, axis="x", lw=1.0, alpha=0.35)
        axa.plot(x[0], y[0], marker="o", ms=4.2, mfc=c, mec="white", mew=0.7, ls="none", zorder=4)
        if m in TOP:
            continue
        off, ha = LBL.get(m, ((-4, 3), "right"))
        lead = dict(arrowstyle="-", color=th.SUB, lw=0.5, shrinkA=1, shrinkB=3) if m in LEADER else None
        axa.annotate(NAME[m], (x[0], y[0]), textcoords="offset points", xytext=off, ha=ha, va="center",
                     fontsize=th.NOTE, color=th.SUB, arrowprops=lead)
    axa.annotate("Opus, GPT-6.1 and Gemma-4-31B\n(hijacked in every episode)", (0.988, 0.985), xycoords="data",
                 xytext=(0.955, 0.83), textcoords="data", ha="right", va="center", fontsize=th.NOTE, color=th.INK,
                 arrowprops=dict(arrowstyle="-", color=th.INK, lw=0.6, shrinkA=2, shrinkB=4))
    from matplotlib.lines import Line2D
    fam = [Line2D([], [], marker="o", ms=4.2, mfc=col, mec="white", ls="none", label=lab)
           for lab, col in (("Claude", th.OVER), ("other closed", "#2F2E2B"), ("open-weight", "#8C8A84"))]
    axa.legend(handles=fam, loc="lower left", fontsize=th.NOTE, frameon=False, handletextpad=0.2,
               borderaxespad=0.2, labelspacing=0.3)
    axa.plot([1.0], [1.0 - T["verdicts_plant"]["final_acc"]], marker="*", ms=9, mfc="none", mew=0.9, color=th.LEAK,
             ls="none", zorder=3)
    axa.annotate("exact Bayes", (1.0, 1.0), textcoords="offset points", xytext=(-2, 7), ha="right", fontsize=th.NOTE,
                 color=th.LEAK)
    axa.text(0.03, 0.97, f"Spearman ρ = {RHO_TXT}", transform=axa.transAxes, ha="left",
             va="top", fontsize=th.NOTE, color=th.SUB)
    axa.text(0.03, 0.62, "the more a model acts like Bayes,\nthe more often it is hijacked", transform=axa.transAxes,
             ha="left", va="top", fontsize=th.NOTE, color=th.INK, style="italic")
    axa.set_xlabel("agreement with the Bayes action on honest chains")
    axa.set_ylabel("share of episodes hijacked\n(all ten genuine agents wrong)")
    axa.set_ylim(-0.05, 1.12)
    # (b) dumbbells, each condition's CI drawn just above / below the row so the two never overlap
    ms = [m for m in EXPLOIT if all(((P.get(m, {}).get(k) or {}).get("pos_acc_genuine") or [0, 0, 0, 0])[3] >= 10
                                    for k in ("signals_plant", "verdicts_plant"))]
    yy = np.arange(len(ms))[::-1].astype(float)
    bv, bs = T["verdicts_plant"]["pos_acc_genuine"], T["signals_plant"]["pos_acc_genuine"]
    axb.axvline(bv, color=th.LEAK, ls=(0, (2, 2)), lw=0.8, alpha=0.8, zorder=1)
    axb.axvline(bs, color=th.SAFE, ls=(0, (2, 2)), lw=0.8, alpha=0.8, zorder=1)
    for yi, m in zip(yy, ms):
        v, s = P[m]["verdicts_plant"]["pos_acc_genuine"], P[m]["signals_plant"]["pos_acc_genuine"]
        axb.plot([v[0], s[0]], [yi, yi], color=th.STONE, lw=0.8, alpha=0.5, zorder=2)
        for val, c, mk, msz, dy in ((v, th.LEAK, "D", 4.0, 0.16), (s, th.SAFE, "o", 4.6, -0.16)):
            axb.plot([val[1], val[2]], [yi + dy, yi + dy], color=c, lw=0.9, alpha=0.6, solid_capstyle="butt",
                     zorder=2)
            axb.plot(val[0], yi, marker=mk, ms=msz, mfc=c, mec="white", mew=0.7, ls="none", zorder=4)
    top = yy[0] + 0.95
    # the two benchmark labels double as the legend: colour matches the markers (diamonds red, circles green)
    axb.text(bv + 0.015, top, f"diamonds:\ntwo fake verdicts\n(rational {bv:.2f})", ha="left", va="center",
             fontsize=th.NOTE, color=th.LEAK)
    axb.text(bs + 0.015, top, f"circles:\ntwo fake spot checks\n(rational {bs:.2f})", ha="left", va="center",
             fontsize=th.NOTE, color=th.SAFE)
    # what an agent gets from its own spot check with no peers at all (p = 0.65 in the task)
    axb.axvline(0.65, color=th.SUB, ls=(0, (1, 2)), lw=0.8, alpha=0.8, zorder=1)
    axb.text(0.665, yy[-1] - 0.75, "own spot check\nalone (0.65)", ha="left", va="center", fontsize=th.NOTE,
             color=th.SUB)
    axb.set_yticks(yy, [NAME.get(m, m) for m in ms])
    axb.set_xlim(-0.06, 1.0)
    axb.set_ylim(yy[-1] - 1.15, yy[0] + 1.6)
    axb.set_xlabel("accuracy of the ten genuine agents")
    axb.grid(False)
    axb.spines["left"].set_visible(False)
    th.panel_labels([axa, axb])
    save(fig, "figs/fig_exploit")



# ------------------------------------------------------------------------------------------------ Fig: protocol
def fig_protocol():
    H = json.load(open(ART / "r3-herding/paper_summary.json", encoding="utf-8"))
    T = H["_theory"]
    th.use_theme()
    fig, (axa, axb) = figure_grid(1, 2, width=th.PAPER_W, ratio=0.38, gridspec_kw={"width_ratios": [1.0, 1.25]})
    ns = sorted(int(n) for n in T)
    axa.plot(ns, [T[str(n)]["pooling_final"] for n in ns], color=th.SAFE, marker="o", ms=2.5, lw=1.1,
             label="share probabilities / signals")
    axa.plot(ns, [T[str(n)]["verdicts_final"] for n in ns], color=th.LEAK, marker="o", ms=2.5, lw=1.1,
             label="share verdicts")
    axa.axvline(12, color=MUTED, lw=0.8, zorder=0)
    axa.text(12.6, 0.705, "our chains", fontsize=th.NOTE, color=BASELINE)
    axa.set_xscale("log", base=2)
    axa.set_xticks(ns, [str(n) for n in ns])
    axa.set_xlabel("number of agents in the chain")
    axa.set_ylabel("final agent accuracy (rational agents)")
    axa.set_ylim(0.69, 1.01)
    axa.legend(loc="upper left", fontsize=th.NOTE, frameon=False)
    # (b) publicly wrong while the private majority was right
    models = [m for m in TIER if m in H and "verdicts" in H[m]]
    yy = np.arange(len(models))[::-1]
    for yi, m in zip(yy, models):
        v = H[m]["verdicts"].get("publicly_wrong_given_private_majority_right")
        alt = H[m].get("probs", {}).get("publicly_wrong_given_private_majority_right")
        if v:
            axb.errorbar(v[0], yi + 0.14, xerr=err(v), fmt="o", ms=2.8, color=th.LEAK, elinewidth=0.6,
                         capsize=0)
        if alt:
            axb.errorbar(alt[0], yi - 0.14, xerr=err(alt), fmt="s", ms=2.6, color=th.SAFE, elinewidth=0.6,
                         capsize=0)
        if v:
            axb.text(1.02, yi, f"n={v[3]}", va="center", ha="left", fontsize=th.NOTE, color=BASELINE,
                     transform=axb.get_yaxis_transform())
    axb.set_yticks(yy, models)
    axb.set_xlim(-0.02, 0.7)
    axb.set_xlabel("final verdict wrong | private majority right (share)")
    axb.errorbar([], [], fmt="o", color=th.LEAK, label="verdicts")
    axb.errorbar([], [], fmt="s", color=th.SAFE, label="probabilities")
    axb.legend(loc="upper right", fontsize=th.NOTE, frameon=True, framealpha=0.95, edgecolor=MUTED)
    th.panel_labels([axa, axb])
    save(fig, "figs/fig_protocol")


if __name__ == "__main__":
    fig_exploit()
    fig_capture()
    fig_herding()
    fig_protocol()
