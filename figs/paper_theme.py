"""Shared visual theme for the paper's data figures, matched to the TikZ schematics (fig_bhw.tex, fig_swap.tex):
Inter (SIL OFL, vendored as TrueType in figs/fonts/), warm neutrals, one hue per meaning, minimal axes.
"""
import pathlib

import matplotlib as mpl
import matplotlib.font_manager as fm

import orx_figstyle as ofs

FONT_DIR = pathlib.Path(__file__).resolve().parent / "fonts"
PAPER_W = 6.5          # \\textwidth of bayesian_swarms.tex (1in margins): build at final size, no rescaling
NOTE = 6.0             # annotation size (pt); Nature: all text 5-7 pt, panel labels 8 pt bold

# palette: identical hex values to the schematics
INK = "#191919"        # text
SUB = "#6B6A66"        # secondary text
AXIS = "#8A877F"       # spines and ticks
GRID = "#ECE8DF"       # gridlines
BAND = "#F5F2EA"       # group bands
LEAK = "#B84A33"       # clay: leak, attack, wrong (5.2:1 on white)
SAFE = "#287F70"       # teal: safe, defence, evidence (4.8:1 on white)
OVER = "#3F6EA8"       # slate blue: frontier / reference
OCHRE = "#C98E1F"      # second warm hue (false alarms)
STONE = "#9A958B"      # neutral models


def use_theme():
    for f in sorted(FONT_DIR.glob("Inter-*.ttf")):
        fm.fontManager.addfont(str(f))
    ofs.use_style()
    ofs._resolved_font = "Inter"
    mpl.rcParams.update({
        "font.sans-serif": ["Inter"] + ofs.SANS,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.labelcolor": INK,
        "axes.labelweight": "medium",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.5,
        "text.color": INK,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": "#4A4844",
        "ytick.labelcolor": "#4A4844",
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 0.0,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.fontsize": 6.5,
        "axes.grid": False,
    })


def soft_errorbar(ax, x, y, ci, color, axis="x", lw=1.6, alpha=0.28, **kw):
    """CI as a soft translucent bar instead of capped whiskers."""
    lo, hi = ci
    if axis == "x":
        ax.plot([lo, hi], [y, y], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=2, **kw)
    else:
        ax.plot([x, x], [lo, hi], color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=2, **kw)


def panel_labels(axes, labels=None):
    """Nature-style panel labels: 8 pt bold lowercase, top left, outside the axes."""
    labels = labels or "abcdefgh"
    for ax, lab in zip(axes, labels):
        ax.text(0.0, 1.0, lab, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold",
                color=INK)
