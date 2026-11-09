"""Regenerate the twin-swap transcripts through the Anthropic API: for each order-fixed SplitFlow pair and each clean
twin (safe = clean0, leaking = clean1), all 8 reviewers write an independent note (round-1 message with an empty
inbox, exactly the frontier_swarm prompt) and give an independent YES/NO verdict. Records are written in the
frontier_swarm cache format (r = 1) so twin_swap.transcripts(<source>, 1) reads them unchanged.

  ANTHROPIC_API_KEY=... python tools/claude_notes.py --model claude-sonnet-5-5 --pairs 60
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import frontier_overseer as fo  # noqa: E402
import frontier_swarm as fs  # noqa: E402
import twin_swap as ts  # noqa: E402


_C = {}


def note_call(model: str, prompt: str):
    """Transcript-writing call with server-side refusal fallback enabled ("default" routing by refusal category).
    Writers of stimuli may differ across notes when a fallback fires; the serving model is recorded."""
    import anthropic
    if "c" not in _C:
        _C["c"] = anthropic.Anthropic(max_retries=4, timeout=600.0)
    try:
        resp = _C["c"].beta.messages.create(
            model=model, max_tokens=4000, output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            messages=[{"role": "user", "content": prompt}])
    except anthropic.APIStatusError as e:
        return {"text": None, "cost": 0.0, "error": f"{e.status_code} {str(e.message)[:160]}", "served": None}
    except anthropic.APIConnectionError as e:
        return {"text": None, "cost": 0.0, "error": f"conn {e!r}"[:200], "served": None}
    if resp.stop_reason == "refusal":
        return {"text": None, "cost": 0.0, "error": "refusal (whole chain)", "served": resp.model}
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {"text": text, "cost": 0.0, "error": None, "served": resp.model}


def main(a):
    fo.MODE["m"] = "fast"
    fs.ORDERED["on"] = True
    src = f"anthropic-api/{a.model}"
    cache = fs.Cache(fs.OUT / f"swarm_{fs.slug(src)}.jsonl")
    eps = [ep for ep in fs.plan(a.pairs) if ep["kind"] in ("clean0", "clean1")]
    tasks = []
    for ep in eps:
        for i in range(fs.N):
            for kind in ("msg", "verdict"):
                k = f"{src}|{ep['id']}|r1|a{i}|{kind}"
                if k not in cache.d:
                    prompt = fs.header(i, ep) + (fs.MSG_INSTR if kind == "msg" else fs.QUESTION + fs.FAST)
                    tasks.append((ep, i, kind, k, prompt))
    print(f"{len(tasks)} calls ({len(cache.d)} cached)", flush=True)

    def work(t):
        ep, i, kind, k, prompt = t
        for _ in range(3):
            res = note_call(a.model, prompt)   # fallback also for verdicts: they only label transcript direction
            txt = (res.get("text") or "").strip()
            verdict = fs.parse(txt) if kind == "verdict" else None
            if txt and (kind == "msg" or verdict in ("YES", "NO")):
                break
        cache.add({"key": k, "ep": ep["id"], "r": 1, "i": i, "kind": kind, "text": txt or None,
                   "cost": res.get("cost", 0.0), "err": res.get("error"), "verdict": verdict,
                   "served": res.get("served", a.model)})

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, tasks))
    print(f"NOTES DONE spend ${fo.SPEND['usd']:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--workers", type=int, default=24)
    main(ap.parse_args())
