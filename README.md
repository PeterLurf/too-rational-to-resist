# Too Rational to Resist: Two Lies Are All It Takes to Hijack Frontier LLMs

Code, data and analysis for the paper [`paper.pdf`](paper.pdf) (Apart Research Swarm Alignment Hackathon,
November 2026).

**Finding in one line.** Frontier LLM agents update on their peers almost exactly as Bayes' rule prescribes, and
that is what makes them exploitable: when agents see only each other's answers, two fake agents answering
first make every genuine agent wrong. Sharing checkable evidence instead of verdicts caps the damage.

## Reproduce in one command

Requires Python ≥ 3.10. No API keys needed.

```bash
git clone https://github.com/PeterLurf/too-rational-to-resist
cd too-rational-to-resist
pip install -r requirements.txt          # or: uv run --with-requirements requirements.txt python reproduce.py
python reproduce.py
```

This rebuilds every table and figure in the paper from the analysis summaries in `data/` (about one minute)
and checks that the regenerated tables are identical to the ones in the paper:

```
Tables identical to the paper's: 5/5
```

### Level 2: re-run every analysis from the raw API logs

```bash
python reproduce.py --from-raw
```

This downloads `raw_api_logs.tar.gz` (19 MB, attached to the
[v1.0 release](https://github.com/PeterLurf/too-rational-to-resist/releases/tag/v1.0)), unpacks it into
`data/`, recomputes every summary from the individual API calls (prompt, model id, full response, parsed
answer, tokens, cost), and then rebuilds the tables and figures. Still no API keys: every call is replayed from
the logs. Takes a few minutes. The capture-test calls for GPT-6.1 and Gemini 4 Argon were run after the v1.0
archive; they are in `raw_addendum/` and merged into `data/` automatically.

### Level 3: re-run the experiments (API keys, costs money)

Set `ANTHROPIC_API_KEY` (Claude), `OPENROUTER_API_KEY` (GPT, Gemini, Qwen, Llama, GPT-OSS, DeepSeek) and, for
Gemma, `ADAPTION_API_KEY` (`pip install adaption==0.19.0`). Calls are cached in `data/`, so with the raw logs
unpacked a rerun only calls the API for missing items; delete a folder's `calls.jsonl` to rerun it from scratch.

| Paper element | Run (Level 3) | Analyse (Level 2) |
|---|---|---|
| Capture test, Fig. 3 (Sec. 4.1) | `tools/twin_swap.py run --mode fast\|reason` (transcripts: `tools/claude_notes.py`) | `tools/twin_swap.py analyse --mode fast\|reason` |
| BHW cascade, Tables 2–3, Figs. 4–6 (Secs. 4.2, 4.5) | `tools/r3_herding.py run --models <aliases>` (see `MODELS` in the file) | `tools/herd_paper.py` |
| Shifted prior (Sec. 4.2) | `tools/r3_herding.py run --models gptcf sonnetcf opuscf haikucf qwen14thinkcf`; `tools/cf_probe.py run` | `tools/cf_analyse.py`, `tools/cf_probe.py analyse` |
| Stress vignettes, Table 5 (Sec. 4.3) | `tools/r3_bayes.py run` | `tools/r3_bayes.py analyse` |
| Hidden profile (Sec. 4.4) | `tools/r3_deaf.py run` | `tools/r3_deaf.py analyse` |
| Rank correlations ρ (Sec. 4.5, Fig. 4a) | `tools/r3_herding.py run` with the `*plant` aliases | `tools/rho_ladder.py` |
| Warned condition (Sec. 4.5) | `tools/r3_herding.py run --models sonnetwarn opuswarn haikuwarn haikuthinkwarn` | `tools/warn_bayes.py analyse` (exact benchmark: `python tools/warn_bayes.py`) |
| Fake-agent sweep, Table 4 | `tools/r3_herding.py run --models sonnetsweep haikusweep haikuthinksweep` | `tools/sweep_analyse.py` |
| Code-review pipeline, Table 1 (Sec. 4.6) | `PR_VARIANT=<folder> tools/pr_review.py run --models … --conds …` (needs the service clone below) | `PR_VARIANT=<folder> tools/pr_review.py analyse` |
| Tables and figures | — | `figs/make_tables_bayes.py`, `figs/make_tab_pr.py`, `figs/make_tab_sweep.py`, `figs/make_figs_bayes.py` |

The code-review experiment uses a read-only clone of a public web service:

```bash
git clone https://github.com/fastapi/full-stack-fastapi-template external/full-stack-fastapi-template
git -C external/full-stack-fastapi-template checkout aabb901
```

Nothing is pushed anywhere and the analytics host in the pull request is fictitious; no request is ever sent.

## Repository layout

```
paper.pdf, paper.tex      the paper (figures in figs/, tables generated into figs/tab_*.tex)
reproduce.py              one-command reproduction (levels 1 and 2)
tools/                    experiment and analysis scripts
figs/                     figure/table generators, TikZ schematics (fig_bhw.tex, fig_swap.tex), Inter font (OFL)
data/                     analysis summaries (JSON), one folder per experiment; the raw logs unpack here
data/stimuli/             the 56 code-snippet pairs used by the capture test
```

All paths default to `data/`; set `SWARM_ARTIFACTS=<dir>` to read and write elsewhere.

## Notes for reviewers

- Refusals, empty and malformed answers are recorded as missing data, never imputed.
- About half of the capture-test transcript notes were served by Claude Sonnet 5 through Anthropic's
  server-side refusal fallback (recorded per call in `data/frontier-swarm/*.jsonl`, field `served`); the
  overseer calls that are measured never use fallbacks. The paper discloses this.
- Pre-registered go/no-go rules, deviations and which analyses are exploratory are listed in Appendix D of the
  paper.
- To recompile the PDF: `python reproduce.py --paper` (needs [tectonic](https://tectonic-typesetting.github.io)),
  or upload `paper.tex` and `figs/` to Overleaf.

## License

Code: MIT (see `LICENSE`). Data and paper: CC BY 4.0. The Inter font in `figs/fonts/` is under the SIL Open
Font License (see `figs/fonts/NOTICE.txt`).
