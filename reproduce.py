"""One-command reproduction for "Too Rational to Resist" (paper.pdf).

    python reproduce.py              # Level 1: rebuild every table and figure from data/ (no API keys, ~1 min)
    python reproduce.py --from-raw   # Level 2: download the raw API logs, re-run every analysis, then Level 1
    python reproduce.py --paper      # also recompile paper.pdf (needs `tectonic` on PATH)

Each level ends with a check that the regenerated tables match the ones in the paper.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_URL = "https://github.com/PeterLurf/too-rational-to-resist/releases/download/v1.0/raw_api_logs.tar.gz"
TABLES = ["tab_herding", "tab_planted", "tab_stress", "tab_sweep", "tab_pr"]
GENERATORS = ["make_tables_bayes.py", "make_tab_pr.py", "make_tab_sweep.py", "make_figs_bayes.py"]
ANALYSES = [  # (description, argv, extra env)
    ("Capture test (Fig. 3)", ["tools/twin_swap.py", "analyse", "--mode", "fast"], {}),
    ("Capture test, reasoning mode", ["tools/twin_swap.py", "analyse", "--mode", "reason"], {}),
    ("BHW cascade + sock-puppets (Tables 2-3, Figs. 4-6)", ["tools/herd_paper.py"], {}),
    ("Rank correlations (Sec. 4.5)", ["tools/rho_ladder.py"], {}),
    ("Warned condition (Sec. 4.5)", ["tools/warn_bayes.py", "analyse"], {}),
    ("Sock-puppet sweep (Table 4)", ["tools/sweep_analyse.py"], {}),
    ("Shifted prior, live chains (Sec. 4.2)", ["tools/cf_analyse.py"], {}),
    ("Shifted prior, probes (Sec. 4.2)", ["tools/cf_probe.py", "analyse"], {}),
    ("Stress vignettes (Table 5)", ["tools/r3_bayes.py", "analyse"], {}),
    ("Hidden profile (Sec. 4.4)", ["tools/r3_deaf.py", "analyse"], {}),
    ("Hidden profile, partition control", ["tools/r3_deaf.py", "analyse", "--out", str(DATA / "r3-deaf" / "partition")], {}),
] + [(f"Code review: {v} (Table 1)", ["tools/pr_review.py", "analyse"], {"PR_VARIANT": v})
     for v in ("haiku", "haiku_ci", "haiku_fq", "haiku_v", "opus", "sonnet", "sonnet_ci", "sonnet_fq", "sonnet_v")]


def run(argv, env=None, quiet=True):
    e = dict(os.environ, SWARM_ARTIFACTS=str(DATA), MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    e.update(env or {})
    r = subprocess.run([sys.executable, *argv], cwd=ROOT, env=e, capture_output=quiet, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode:
        print(r.stdout[-2000:] if r.stdout else "", r.stderr[-2000:] if r.stderr else "")
        sys.exit(f"FAILED: {' '.join(argv)}")


def ensure_raw():
    if any((DATA / "r3-herding").glob("*/calls.jsonl")):
        return
    arc = ROOT / "raw_api_logs.tar.gz"
    if not arc.exists():
        print(f"downloading raw API logs (19 MB) from {RAW_URL}")
        urllib.request.urlretrieve(RAW_URL, arc)
    print("unpacking raw API logs into data/")
    with tarfile.open(arc) as t:
        t.extractall(DATA, filter="data") if sys.version_info >= (3, 12) else t.extractall(DATA)
    merge_addendum()


def merge_addendum():
    """Add the capture-test calls run after the v1.0 archive (GPT-6.1 and Gemini 4 Argon, raw_addendum/) to the
    twin-swap response caches, skipping any key already present."""
    import json
    for mode in ("fast", "reason"):
        extra = ROOT / "raw_addendum" / f"twin_swap_responses_{mode}.jsonl"
        cache = DATA / "twin-swap" / f"responses_{mode}.jsonl"
        if not extra.exists():
            continue
        have = set()
        if cache.exists():
            have = {json.loads(l)["key"] for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()}
        new = [l for l in extra.read_text(encoding="utf-8").splitlines() if l.strip() and json.loads(l)["key"] not in have]
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "a", encoding="utf-8") as fh:
            fh.writelines(l + "\n" for l in new)
        print(f"merged {len(new)} addendum calls into {cache.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-raw", action="store_true", help="re-run every analysis from the raw API logs first")
    ap.add_argument("--paper", action="store_true", help="recompile paper.pdf with tectonic")
    a = ap.parse_args()

    before = {t: (ROOT / "figs" / f"{t}.tex").read_text(encoding="utf-8") for t in TABLES}
    if a.from_raw:
        ensure_raw()
        for desc, argv, env in ANALYSES:
            print(f"[analysis] {desc}")
            run(argv, env)
    for g in GENERATORS:
        print(f"[figures/tables] figs/{g}")
        run([f"figs/{g}"])
    same = [t for t in TABLES if (ROOT / "figs" / f"{t}.tex").read_text(encoding="utf-8") == before[t]]
    diff = [t for t in TABLES if t not in same]
    print(f"\nTables identical to the paper's: {len(same)}/{len(TABLES)}" + (f"  (differ: {diff})" if diff else ""))
    print("Figures written to figs/fig_*.pdf and figs/fig_*.svg (open them next to paper.pdf).")
    if a.paper:
        tec = shutil.which("tectonic")
        if not tec:
            sys.exit("tectonic not found: install it from https://tectonic-typesetting.github.io or upload paper.tex "
                     "and figs/ to Overleaf.")
        for f in ("fig_bhw", "fig_swap"):
            subprocess.run([tec, f"figs/{f}.tex", "--outdir", "figs"], cwd=ROOT, check=True)
        subprocess.run([tec, "paper.tex"], cwd=ROOT, check=True)
        print("paper.pdf rebuilt")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
