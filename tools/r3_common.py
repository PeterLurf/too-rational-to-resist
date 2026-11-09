"""Shared helpers for the round-3 API pilots (Anthropic API only).

- call(model, prompt, mode): one request via the Anthropic Python SDK.
    mode "fast":  lightest setting (Haiku: thinking disabled; Sonnet: thinking between_tools; Opus: effort low,
                  thinking cannot be disabled on Opus 5.5), max_tokens 4000.
    mode "think": adaptive thinking (the default) at effort "medium", max_tokens 16000.
  Returns {"text", "in_tok", "out_tok", "cost", "stop", "error"}. out_tok includes thinking tokens (billed), so it is
  the effort proxy. No sampling parameters (rejected on these models). No server-side fallbacks: a refusal is returned
  as error="refusal <category>" and must be treated as MISSING data, never as an answer.
- DRYRUN=1 in the environment makes call() return a random YES/NO without any network call (for smoke tests).
- Cache: append-only JSONL keyed by a string; rerunning skips keys already answered.
- run_tasks(tasks, fn, workers): thread-pool runner with progress lines and a global budget stop.
- boot(vals, groups): mean with a 95% cluster-bootstrap CI over groups -> [mean, lo, hi, n].
- parse_yes_no(text): last 'ANSWER: YES|NO' if present, else the last bare YES/NO token.
The API key is read by the SDK from ANTHROPIC_API_KEY in the process environment; never print it.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import threading
import time

import numpy as np

PRICE = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}
SPEND = {"usd": 0.0}
_LOCK = threading.Lock()
_CLIENT = {}


def call(model: str, prompt: str, mode: str = "think", max_tokens: int | None = None) -> dict:
    if os.environ.get("DRYRUN") == "1":
        ans = random.choice(["YES", "NO"])
        return {"text": f"(dry run)\nANSWER: {ans}", "in_tok": len(prompt) // 4, "out_tok": random.randint(20, 400),
                "cost": 0.0, "stop": "end_turn", "error": None}
    if model.startswith("adaption:"):    # Adaption Labs open-weight inference; key from ADAPTION_API_KEY
        return _call_adaption(model.split(":", 1)[1], prompt, mode, max_tokens)
    if "/" in model:                     # OpenRouter model id (e.g. openai/gpt-6.1-sol); key from OPENROUTER_API_KEY
        return _call_openrouter(model, prompt, mode, max_tokens)
    import anthropic
    if "c" not in _CLIENT:
        _CLIENT["c"] = anthropic.Anthropic(max_retries=6, timeout=600.0)
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if mode == "fast":
        kw["max_tokens"] = max_tokens or 4000
        kw["output_config"] = {"effort": "low"}
        if model == "claude-sonnet-5-5":
            kw["thinking"] = {"type": "between_tools"}
        elif model == "claude-haiku-5-5":
            kw["thinking"] = {"type": "disabled"}
    else:
        kw["max_tokens"] = max_tokens or 16000
        kw["output_config"] = {"effort": "medium"}
    try:
        resp = _CLIENT["c"].messages.create(**kw)
    except anthropic.BadRequestError as e:
        return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None, "error": f"400 {str(e.message)[:200]}"}
    except anthropic.RateLimitError as e:
        return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None, "error": f"429 {str(e.message)[:120]}"}
    except anthropic.APIStatusError as e:
        return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None,
                "error": f"{e.status_code} {str(e.message)[:160]}"}
    except anthropic.APIConnectionError as e:
        return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None, "error": f"conn {e!r}"[:200]}
    u = resp.usage
    pin, pout = PRICE.get(model, (0.0, 0.0))
    cost = (u.input_tokens * pin + u.output_tokens * pout) / 1e6
    with _LOCK:
        SPEND["usd"] += cost
    if resp.stop_reason == "refusal":
        cat = getattr(resp.stop_details, "category", None) if resp.stop_details else None
        return {"text": None, "in_tok": u.input_tokens, "out_tok": u.output_tokens, "cost": cost, "stop": "refusal",
                "error": f"refusal {cat}"}
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {"text": text, "in_tok": u.input_tokens, "out_tok": u.output_tokens, "cost": cost,
            "stop": resp.stop_reason, "error": None}



NO_REASONING_PARAM = {"meta-llama/llama-3.1-8b-instruct", "google/gemma-3-12b-it", "meta-llama/llama-3.3-70b-instruct"}


def _call_openrouter(model: str, prompt: str, mode: str, max_tokens: int | None) -> dict:
    """OpenRouter backend for non-Claude models. think -> reasoning effort medium; fast -> minimal. No temperature."""
    import requests
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None, "error": "no OPENROUTER_API_KEY"}
    if mode == "fast" and model.startswith("qwen/qwen3"):
        prompt = prompt + "\n/no_think"                       # Qwen3 soft switch; reasoning.enabled=false alone is ignored
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or (4000 if mode == "fast" else 16000), "usage": {"include": True}}
    if model in NO_REASONING_PARAM:
        pass                                                   # model has no reasoning mode: send no reasoning field
    elif mode == "fast" and model.startswith("qwen/qwen3"):
        body["reasoning"] = {"enabled": False}
    else:
        body["reasoning"] = {"effort": "minimal" if mode == "fast" else "medium"}
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "swarm-r3-pilots"}
    last = None
    for attempt in range(5):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=hdr, json=body, timeout=600)
            if r.status_code in (402, 403):        # out of credit / key limit: stop, do not retry
                return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None,
                        "error": f"{r.status_code} key limit or credit"}
            if r.status_code in (408, 429, 500, 502, 503, 504):
                last = f"{r.status_code}"
                time.sleep(min(60, 4 * 2 ** attempt) + random.random())
                continue
            r.raise_for_status()
            j = r.json()
            ch = (j.get("choices") or [{}])[0]
            u = j.get("usage") or {}
            cost = float(u.get("cost") or 0.0)
            with _LOCK:
                SPEND["usd"] += cost
            return {"text": (ch.get("message") or {}).get("content") or "", "in_tok": u.get("prompt_tokens") or 0,
                    "out_tok": u.get("completion_tokens") or 0, "cost": cost, "stop": ch.get("finish_reason"),
                    "error": None}
        except Exception as ex:
            last = repr(ex)[:200]
            time.sleep(min(60, 4 * 2 ** attempt) + random.random())
    return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None, "error": last}

def _call_adaption(model: str, prompt: str, mode: str, max_tokens: int | None) -> dict:
    """Adaption Labs backend (OpenAI Responses-style SDK, adaption==0.19.0, default base URL). The SDK has no reasoning
    switch, so each model runs in its default mode (recorded as mode "default"). usage.cost is in Adaption credits
    (USD-like scale) and is added to SPEND. Errors and empty outputs are returned as error=..., i.e. MISSING data."""
    try:
        from adaption import Adaption
    except Exception as ex:  # package missing
        return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None, "error": f"import {ex!r}"[:200]}
    if "adaption" not in _CLIENT:
        _CLIENT["adaption"] = Adaption(timeout=300.0, max_retries=0)  # key from ADAPTION_API_KEY; retries below
    client = _CLIENT["adaption"]
    last = None
    for attempt in range(9):
        try:
            r = client.responses.create(model=model, input=prompt, temperature=0,
                                        max_output_tokens=max_tokens or (4000 if mode == "fast" else 16000))
            u = r.usage
            cost = float(getattr(u, "cost", 0.0) or 0.0)
            with _LOCK:
                SPEND["usd"] += cost
            text = r.output_text or ""
            det = getattr(u, "output_tokens_details", None)
            return {"text": text, "in_tok": getattr(u, "input_tokens", 0) or 0,
                    "out_tok": getattr(u, "output_tokens", 0) or 0, "cost": cost,
                    "stop": getattr(r, "status", None), "error": None if text.strip() else "empty output",
                    "rtok": getattr(det, "reasoning_tokens", None) if det else None}
        except Exception as ex:
            code = getattr(ex, "status_code", None) or getattr(getattr(ex, "response", None), "status_code", None)
            last = f"{code} {ex!r}"[:240]
            if code is not None and code not in (408, 409, 429, 500, 502, 503, 504):
                break                                  # client error: do not retry
            if code == 429:                            # per-user limit: "retry after 60 seconds"
                time.sleep(65 + 10 * random.random())
            elif attempt >= 4:
                break
            else:
                time.sleep(min(60, 4 * 2 ** attempt) + random.random())
    return {"text": None, "in_tok": 0, "out_tok": 0, "cost": 0.0, "stop": None, "error": last}


_ANS = re.compile(r"ANSWER\s*[:\-]?\s*\**\s*(YES|NO)\b", re.I)


def parse_yes_no(text: str | None):
    if not text:
        return None
    m = _ANS.findall(text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b(YES|NO)\b", text.upper())
    return m[-1] if m else None


class Cache:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock, self.d = threading.Lock(), {}
        if self.path.exists():
            for line in open(self.path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok"):
                    self.d[r["key"]] = r

    def add(self, rec: dict):
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if rec.get("ok"):
                self.d[rec["key"]] = rec


def run_tasks(tasks, fn, workers: int = 3, budget: float = 30.0, label: str = ""):
    """tasks: list of task objects; fn(task) -> record (performs the call and caches). Stops submitting new work once
    the global spend exceeds the budget."""
    t0, n = time.time(), 0
    stop = threading.Event()

    def guarded(t):
        if stop.is_set():
            return None
        r = fn(t)
        if SPEND["usd"] > budget:
            stop.set()
        return r

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(guarded, tasks):
            n += 1
            if n % 50 == 0 or n == len(tasks):
                print(f"[{time.strftime('%H:%M:%S')}] {label} {n}/{len(tasks)} spend ${SPEND['usd']:.2f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if stop.is_set():
        print(f"BUDGET STOP at ${SPEND['usd']:.2f}", flush=True)


def boot(vals, groups, nb: int = 2000, seed: int = 0):
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ok = ~np.isnan(vals)
    vals, groups = vals[ok], groups[ok]
    if len(vals) == 0:
        return None
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    bs = [vals[np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])].mean() for _ in range(nb)]
    return [round(float(vals.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4), int(len(vals))]
