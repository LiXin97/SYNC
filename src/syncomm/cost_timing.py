"""Wall-clock timers for the cost axis (item 10, timing half; 2026-09-12).

Additive instrumentation only: every function here is called by the runners beside the
work they already do and writes extra keys; nothing here changes a logit, a packet or a
row that existed before. Rules (the study protocol section 3.3):
`torch.cuda.synchronize()` immediately before start and before stop of every timed
region; `perf_counter` for wall; batch = 1; the first WARMUP rows of a process are
reported separately (cold) and excluded from the summary.
"""
from __future__ import annotations

import os
import platform
import socket
import statistics
import time

import torch

WARMUP = 3           # rows excluded from every summary (CUDA context, autotune, page-in)

perf = time.perf_counter     # named alias: instrumentation added after a frozen snapshot must be
                             # identifiable BY NAME so the code-drift guard can neutralise it


def sync(t=None) -> None:
    """Synchronise before starting or stopping a timed region. `t` may be a tensor, in which case
    the synchronisation is a no-op unless that tensor is on CUDA."""
    if t is not None:
        if getattr(t, "is_cuda", False):
            torch.cuda.synchronize(t.device)
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """`with Timer() as t: ...; t.ms` -- synchronised wall-clock milliseconds."""

    def __enter__(self):
        sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync()
        self.ms = (time.perf_counter() - self.t0) * 1000.0
        return False


def environment() -> dict:
    d = {"hostname": socket.gethostname(), "python": platform.python_version(),
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
         "utc": time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())}
    try:
        d["container_start"] = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime(os.stat("/proc/1").st_ctime))
    except Exception:
        pass
    return d


def text_gen_prompt(passage: str, budget: int) -> str:
    """Two-shot budgeted-note prompt for the sender-decode cost of text_gen@B (base models
    continue a note; the smoke of 2026-09-12 showed a bare "Message:" prompt yields one
    token). The wording does not matter for cost; the decode length is capped exactly as
    run_j3final.rd_gen_max_tokens caps it (max(8, B // 2) new tokens) and, as there, the
    generation runs to the cap or EOS and is truncated to the budget afterwards."""
    return (f"Compress each passage into a note of at most {budget} bytes.\n\n"
            "Passage: The Eiffel Tower stands in Paris and was completed in 1889 for the World's Fair.\n"
            f"Note: {_trunc('Eiffel Tower, Paris, 1889', budget)}\n\n"
            "Passage: Mount Fuji is the highest mountain in Japan, rising 3776 m above sea level.\n"
            f"Note: {_trunc('Fuji, Japan, 3776 m', budget)}\n\n"
            f"Passage: {passage}\nNote:")


def _trunc(s: str, budget: int) -> str:
    cur = ""
    for w in s.split():
        cand = f"{cur} {w}" if cur else w
        if len(cand.encode("utf-8")) > budget:
            break
        cur = cand
    return cur


def gen_max_tokens(budget: int) -> int:
    return max(8, int(budget) // 2)          # == run_j3final.rd_gen_max_tokens


@torch.no_grad()
def time_text_gen(model, tok, passage: str, budget: int, dev) -> dict:
    """Greedy generation at the packet's byte budget; returns seconds, new tokens,
    prompt tokens and the truncated message's byte length (never the message)."""
    prompt = text_gen_prompt(passage, budget)
    enc = tok(prompt, return_tensors="pt")
    ids = enc.input_ids.to(dev)
    mask = enc.attention_mask.to(dev)
    with Timer() as t:
        out = model.generate(ids, attention_mask=mask, max_new_tokens=gen_max_tokens(budget),
                             do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    first = raw.strip().split("\n")[0]
    msg, cur = "", ""
    for w in first.split():
        cand = f"{cur} {w}" if cur else w
        if len(cand.encode("utf-8")) > budget:
            break
        cur = cand
    msg = cur
    return {"textgen_budget_bytes": int(budget), "textgen_max_new_tokens": gen_max_tokens(budget),
            "textgen_ms": round(t.ms, 3), "textgen_new_tokens": int(out.shape[1] - ids.shape[1]),
            "textgen_prompt_tokens": int(ids.shape[1]),
            "textgen_msg_bytes": len(msg.encode("utf-8"))}


def summarize(values, warmup: int = WARMUP) -> dict:
    """median / p5 / p95 / n over the rows AFTER the warmup ones; cold rows listed."""
    vals = [float(v) for v in values if v is not None]
    cold, warm = vals[:warmup], vals[warmup:]
    if not warm:
        return {"n": 0, "cold": cold}
    s = sorted(warm)
    q = lambda p: s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    return {"n": len(warm), "median": round(statistics.median(warm), 4), "p5": round(q(0.05), 4),
            "p95": round(q(0.95), 4), "mean": round(statistics.fmean(warm), 4), "cold": [round(c, 4) for c in cold]}
