"""B0 episode builder: OpenBookQA / SciQ / ARC-Challenge -> SYNC episode JSONL.

Spec: ``the study protocol`` §1.1 (benchmark set + roles), §1.2
(information-asymmetric split protocol and the Hard-Rule-9 leakage screens),
§5 item 4 (split builder + leakage checker is a first-48h deliverable), and the
``PI AMENDMENT 2026-08-05`` (three committed standard benchmarks: OBQA headline,
SciQ committed second, ARC-Challenge promoted from gated satellite).

EPISODE SCHEMA (one JSON object per line, key order stable)::

    item_id          str   "<benchmark>/<split>/<native id or index>"
    benchmark        str   obqa | sciq | arc-c
    split            str   the upstream split name, verbatim
    sender_evidence  str   the WITHHELD text only the sender sees:
                             obqa  -> official gold science fact (`fact1`)
                             sciq  -> the native per-item `support` passage
                             arc-c -> ARC Corpus BM25 top-3  [STUB in B0]
    evidence_source  str   provenance tag for the line above
    question_stem    str   receiver-visible question
    choices          list  [{"label": "A", "text": ...}, ...] always 4, A-D
    answerKey        str   one of A B C D
    leakage_flags    dict  see `screen_leakage` (§1.2); logged, never silently
                           dropped -- drops require the explicit --drop-violations
    item_sha8        str   sha256(question + choices + answerKey)[:8]

WHAT THIS FILE DELIBERATELY DOES NOT DO
---------------------------------------
* No packet machinery, no model, no torch import: B0 runs on the CPU dev box
  with zero node contact (§3, row "08-04 [non-English note removed]–08-06").
* No thresholding of the logged overlap ratios. §1.2/PI-amendment say the SciQ
  support<->question overlap is *logged per item*; the number that turns it into
  a drop rule belongs to `gates/benchmark_dev.lock.json` (08-07 signature), and
  writing a threshold here before that lock is signed would violate Hard Rule 1.

MEASURED SCREEN RATES (dev box, cached HF data, this file, 2026-08-04 — quoted
so the 08-07 dev lock picks its screen from numbers rather than from intuition;
recompute with `--split validation` before citing anywhere else, Hard Rule 2):

    obqa/validation  n=500  gold_verbatim  44  distractor_verbatim  12
                            answer_letter   0  violation_strict     55 (11.0%)
                            overlap ratio  p50 0.333  p90 1.000
    sciq/validation  n=887  gold_verbatim 813  distractor_verbatim  65
                            answer_letter   0  violation_strict    814 (91.8%)
                            overlap ratio  p50 0.875  p90 1.000
    arc-c/validation n=295  all zero (evidence is the empty STUB), 4 items
                            dropped for a choice count != 4

The SciQ line is the decision this file refuses to make silently: `strict`
(spec-literal, §1.2) would void 92% of SciQ because a support passage that does
not state the answer is not a support passage. `option_set` voids 7.3%. Both
numbers are logged for every item under both definitions; `--screen` records
which one an artifact was built under.

DETERMINISM
-----------
Upstream row order is preserved (``datasets`` iteration is deterministic), no
RNG is used anywhere, and the one derived ordering -- the SciQ choice order,
which the upstream data does not provide -- is a pure function of the question
text (`gold_position`). Two runs of the same command produce byte-identical
output; `tests/test_bench_harness.py` pins this.

USAGE::

    PYTHONPATH=src python3 -m syncomm.bench.build_bench \\
        --benchmark obqa --split test --out data/bench/obqa_test.jsonl
    PYTHONPATH=src python3 -m syncomm.bench.build_bench \\
        --benchmark sciq --split validation --out data/bench/sciq_validation.jsonl
    PYTHONPATH=src python3 -m syncomm.bench.build_bench \\
        --benchmark arc-c --split validation --arc-evidence stub \\
        --allow-stub-evidence --out data/bench/arc-c_validation.jsonl
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys

# --------------------------------------------------------------------------
# Upstream identities. Pinned here so a rebuild cannot silently swap config
# (the OBQA "additional" config is the ONLY one carrying `fact1`, which is what
# makes the §1.1 information-asymmetric split canonical rather than home-made).
# --------------------------------------------------------------------------
SOURCES = {
    "obqa": {"hf": "allenai/openbookqa", "config": "additional",
             "evidence_source": "obqa_fact1"},
    "sciq": {"hf": "allenai/sciq", "config": None,
             "evidence_source": "sciq_support"},
    "arc-c": {"hf": "allenai/ai2_arc", "config": "ARC-Challenge",
              "evidence_source": "arc_corpus_bm25_top3"},
}

LABELS = ("A", "B", "C", "D")
N_CHOICES = 4

# Upstream ARC/OBQA label alphabets that must be mapped onto A-D. ARC items use
# either letters or 1-4; anything else (or a choice count != 4) is dropped and
# counted, never silently re-labelled.
NUMERIC_LABELS = {"1": "A", "2": "B", "3": "C", "4": "D"}

# §1.2 Hard Rule 9 screen 2: "[non-English note removed] message [non-English note removed]".
# Two independent patterns, both anchored so ordinary prose ("vitamin D.",
# "part A of the cell") does not fire:
#   (a) an explicit answer declaration  -> "answer: B", "the answer is (C)"
#   (b) an option enumeration           -> "(A)" or a line-initial "B."
_ANSWER_DECL_RE = re.compile(
    r"\banswer\b[^A-Za-z0-9]{0,12}\(?([A-D])\)?(?![A-Za-z0-9])", re.IGNORECASE)
_ANSWER_ENUM_RE = re.compile(r"\(([A-D])\)|(?:^|\n)\s*([A-D])[.)](?=\s)")

_WORD_RE = re.compile(r"[a-z0-9]+")

# Stopwords for the overlap ratio only (§1.2 logged diagnostic). Deliberately a
# small closed list -- no nltk dependency on the dev box, and a frozen list is
# reproducible where a library's list drifts across versions.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with without into over under about as is are was were be been being am do
does did done has have had having it its it's they them their he she his her
we us our you your i me my not no nor so such can could will would shall should
may might must what which who whom whose when where why how all any both each
few more most other some only own same too very s t don now
""".split())


# ------------------------------------------------------------ tiny utilities

def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H%M%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(s: str) -> str:
    """Lowercase, punctuation -> space, whitespace collapsed, space padded.

    Padding both ends lets a plain substring test act as a word-boundary test,
    so the verbatim screen does not fire on "ice" inside "device" (§1.2).
    """
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower())
    return " " + " ".join(s.split()) + " "


def contains_verbatim(haystack_norm: str, needle: str) -> bool:
    n = norm_text(needle)
    if n.strip() == "":
        return False
    return n in haystack_norm


def content_words(s: str) -> set:
    """Lowercased alphanumeric tokens minus STOPWORDS, crude plural stripping.

    Used ONLY for the logged evidence<->question overlap ratio (§1.2 / PI
    amendment). It is a diagnostic, not a gate: no threshold is applied here.
    """
    out = set()
    for w in _WORD_RE.findall((s or "").lower()):
        if w in STOPWORDS:
            continue
        if len(w) > 4 and w.endswith("es") and w[:-2].endswith(
                ("s", "x", "z", "ch", "sh")):
            w = w[:-2]                       # boxes -> box, dishes -> dish
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                       # magnets -> magnet
        out.add(w)
    return out


def item_sha8(question: str, choices: list, answer_key: str) -> str:
    payload = json.dumps({"q": question,
                          "c": [[c["label"], c["text"]] for c in choices],
                          "a": answer_key}, sort_keys=True, ensure_ascii=False)
    return sha256_bytes(payload.encode("utf-8"))[:8]


# --------------------------------------------------------------- the screens

def screen_leakage(evidence: str, choices: list, answer_key: str,
                   question: str) -> dict:
    """The §1.2 (Hard Rule 9) leakage screens, computed per item.

    Returned flags:

    ``choices_verbatim``      labels whose choice text occurs verbatim
                              (normalised, word-bounded) inside the evidence.
    ``gold_verbatim``         the answerKey's text occurs in the evidence. This
                              is the INTENDED information asymmetry for SciQ
                              (the support passage states the answer) -- it is
                              reported separately precisely so the dev lock can
                              distinguish "evidence entails the answer" from
                              "evidence exposes the option set".
    ``distractors_verbatim``  non-gold option texts in the evidence. These mean
                              the *option set* leaked backwards into the sender's
                              view, which §1.2 forbids ("sender [non-English note removed]
                              question/[non-English note removed]/[non-English note removed]").
    ``answer_letter_pattern`` the evidence contains an answer-letter pattern.
    ``evidence_question_overlap``  |Q ∩ E| / |Q| over content words -- the SciQ
                              support<->question overlap RATIO the PI amendment
                              requires logged per item. LOGGED, NOT THRESHOLDED
                              (the threshold belongs to the 08-07 dev lock).
    ``evidence_question_jaccard``  same sets, Jaccard, for the lock's convenience.
    ``violation_strict``      any choice text verbatim OR an answer letter.
    ``violation_option_set``  a DISTRACTOR verbatim OR an answer letter.

    ``violation`` itself is set by the caller from ``--screen`` so that which of
    the two definitions is operative is a recorded, dated choice rather than an
    implicit one.
    """
    ev_norm = norm_text(evidence)
    verbatim = [c["label"] for c in choices
                if contains_verbatim(ev_norm, c["text"])]
    distract = [l for l in verbatim if l != answer_key]
    letter = bool(_ANSWER_DECL_RE.search(evidence or "")
                  or _ANSWER_ENUM_RE.search(evidence or ""))
    q_words, e_words = content_words(question), content_words(evidence)
    inter = q_words & e_words
    union = q_words | e_words
    return {
        "choices_verbatim": verbatim,
        "gold_verbatim": answer_key in verbatim,
        "distractors_verbatim": distract,
        "answer_letter_pattern": letter,
        "evidence_empty": not (evidence or "").strip(),
        "evidence_question_overlap": (round(len(inter) / len(q_words), 4)
                                      if q_words else 0.0),
        "evidence_question_jaccard": (round(len(inter) / len(union), 4)
                                      if union else 0.0),
        "violation_strict": bool(verbatim) or letter,
        "violation_option_set": bool(distract) or letter,
    }


def choices_from_labels(labels, texts) -> list | None:
    """Normalise an upstream (labels, texts) pair onto A-D, or None if it cannot
    be normalised (wrong arity or an unknown label alphabet -> dropped+counted).
    """
    if len(labels) != N_CHOICES or len(texts) != N_CHOICES:
        return None
    mapped = []
    for l in labels:
        l = str(l).strip()
        l = NUMERIC_LABELS.get(l, l.upper())
        if l not in LABELS:
            return None
        mapped.append(l)
    if sorted(mapped) != list(LABELS):
        return None
    by_label = {l: t for l, t in zip(mapped, texts)}
    return [{"label": l, "text": str(by_label[l]).strip()} for l in LABELS]


def gold_position(question: str) -> int:
    """Deterministic gold slot for benchmarks whose options are unordered (SciQ).

    A hash of the question text, not an RNG: rebuilding the split on any machine
    at any time gives the same file. Uniformity over the four slots is asserted
    downstream by `assert_choice_balance` (the run_j2 ADDENDUM2 position-confound
    guard, carried over per §1.2).
    """
    return int(hashlib.sha256(question.encode("utf-8")).hexdigest()[:8], 16) % N_CHOICES


def assert_choice_balance(episodes: list, min_n: int = 100,
                          tol: float = 0.10) -> dict:
    """Answer-position distribution + the ADDENDUM2-style confound guard.

    run_j2/run_demo abort when the gold option sits in one position far more
    often than chance; the same failure mode (a receiver that answers from
    position prior alone) applies verbatim to a 4-way MC readout. Enforced only
    at n >= min_n so that 5-item smoke builds stay usable; below that the
    fractions are reported and not enforced.
    """
    n = len(episodes)
    frac = {l: 0.0 for l in LABELS}
    for ep in episodes:
        frac[ep["answerKey"]] += 1.0 / max(n, 1)
    worst = max(abs(frac[l] - 1.0 / N_CHOICES) for l in LABELS) if n else 0.0
    if n >= min_n and worst > tol:
        raise SystemExit(
            f"ABORT: answer-position confound, worst |p-0.25| = {worst:.3f} > "
            f"{tol} over n={n} (run_j2 ADDENDUM2 guard, carried to the MC "
            f"readout by BENCH_PLAN §1.2). Fractions: "
            + ", ".join(f"{l}={frac[l]:.3f}" for l in LABELS))
    return {"n": n, "fractions": {l: round(frac[l], 4) for l in LABELS},
            "worst_abs_dev": round(worst, 4), "enforced": n >= min_n}


# ------------------------------------------------------------- the builders

def _load_hf(benchmark: str, split: str):
    from datasets import load_dataset          # imported lazily: B0 is CPU-only
    src = SOURCES[benchmark]
    ds = (load_dataset(src["hf"], src["config"]) if src["config"]
          else load_dataset(src["hf"]))
    if split not in ds:
        raise SystemExit(f"ABORT: split {split!r} not in {src['hf']} "
                         f"(have {sorted(ds)})")
    return ds[split]


def build_obqa(rows, split: str, drops: dict) -> list:
    """OBQA `additional`: sender evidence = the official per-item gold fact1
    (§1.1, "[non-English note removed] per-item gold fact [non-English note removed] MC [non-English note removed]")."""
    out = []
    for r in rows:
        ch = choices_from_labels(r["choices"]["label"], r["choices"]["text"])
        if ch is None:
            drops["bad_choices"] = drops.get("bad_choices", 0) + 1
            continue
        key = str(r["answerKey"]).strip().upper()
        key = NUMERIC_LABELS.get(key, key)
        if key not in LABELS:
            drops["bad_answer_key"] = drops.get("bad_answer_key", 0) + 1
            continue
        out.append({
            "item_id": f"obqa/{split}/{r['id']}",
            "sender_evidence": (r.get("fact1") or "").strip(),
            "question_stem": r["question_stem"].strip(),
            "choices": ch, "answerKey": key,
        })
    return out


def build_sciq(rows, split: str, drops: dict, require_support: bool) -> list:
    """SciQ: sender evidence = the native per-item `support` passage.

    The upstream item is (question, correct_answer, distractor1..3) with NO
    option order, so the order is derived deterministically from the question
    hash (`gold_position`) and the balance is asserted afterwards.
    """
    out = []
    for i, r in enumerate(rows):
        support = (r.get("support") or "").strip()
        if require_support and not support:
            drops["empty_support"] = drops.get("empty_support", 0) + 1
            continue
        q = (r["question"] or "").strip()
        distractors = [(r.get("distractor1") or "").strip(),
                       (r.get("distractor2") or "").strip(),
                       (r.get("distractor3") or "").strip()]
        gold = (r.get("correct_answer") or "").strip()
        if not gold or any(not d for d in distractors):
            drops["missing_option_text"] = drops.get("missing_option_text", 0) + 1
            continue
        g = gold_position(q)
        texts, di = [], 0
        for slot in range(N_CHOICES):
            if slot == g:
                texts.append(gold)
            else:
                texts.append(distractors[di])
                di += 1
        ch = choices_from_labels(list(LABELS), texts)
        if ch is None:                                   # unreachable by arity
            drops["bad_choices"] = drops.get("bad_choices", 0) + 1
            continue
        out.append({
            "item_id": f"sciq/{split}/{i:05d}",
            "sender_evidence": support,
            "question_stem": q, "choices": ch, "answerKey": LABELS[g],
        })
    return out


def build_arc(rows, split: str, drops: dict, evidence_mode: str,
              corpus: str | None) -> list:
    """ARC-Challenge. Sender evidence needs ARC-Corpus BM25 top-3 (§1.1 /
    PI amendment item 3) and the corpus is NOT on disk yet (§5 item 1 lists its
    acquisition as a pending first-48h task), so B0 ships the retrieval behind a
    flag:

      --arc-evidence stub  -> empty evidence, `evidence_source` tagged _STUB,
                              `--allow-stub-evidence` required to write the file
      --arc-evidence bm25  -> not implemented in B0; aborts loudly rather than
                              quietly returning something weaker (Hard Rule 10)
    """
    if evidence_mode == "bm25":
        raise SystemExit(
            "ABORT: --arc-evidence bm25 is NOT IMPLEMENTED in B0. The ARC "
            "Corpus is not on disk (BENCH_PLAN §5 item 1: acquisition + "
            "checksum is a pending first-48h task) and a retrieval this claim "
            "rests on will not be faked. Re-run with --arc-evidence stub to "
            "build the receiver-side fields only, or land the corpus first."
            + (f" (--arc-corpus was {corpus!r})" if corpus else ""))
    out = []
    for r in rows:
        ch = choices_from_labels(r["choices"]["label"], r["choices"]["text"])
        if ch is None:
            drops["bad_choices"] = drops.get("bad_choices", 0) + 1
            continue
        key = str(r["answerKey"]).strip().upper()
        key = NUMERIC_LABELS.get(key, key)
        if key not in LABELS:
            drops["bad_answer_key"] = drops.get("bad_answer_key", 0) + 1
            continue
        out.append({
            "item_id": f"arc-c/{split}/{r['id']}",
            "sender_evidence": "",
            "question_stem": (r.get("question") or "").strip(),
            "choices": ch, "answerKey": key,
        })
    return out


def to_episode(raw: dict, benchmark: str, split: str, evidence_source: str,
               screen: str) -> dict:
    flags = screen_leakage(raw["sender_evidence"], raw["choices"],
                           raw["answerKey"], raw["question_stem"])
    flags["screen"] = screen
    flags["violation"] = flags[f"violation_{screen}"]
    return {
        "item_id": raw["item_id"],
        "benchmark": benchmark,
        "split": split,
        "sender_evidence": raw["sender_evidence"],
        "evidence_source": evidence_source,
        "question_stem": raw["question_stem"],
        "choices": raw["choices"],
        "answerKey": raw["answerKey"],
        "leakage_flags": flags,
        "item_sha8": item_sha8(raw["question_stem"], raw["choices"],
                               raw["answerKey"]),
    }


def build_episodes(benchmark: str, split: str, *, limit: int | None = None,
                   screen: str = "strict", require_support: str = "auto",
                   arc_evidence: str = "stub", arc_corpus: str | None = None,
                   rows=None) -> tuple[list, dict]:
    """-> (episodes, stats). `rows` overrides the HF load (tests, fixtures)."""
    if benchmark not in SOURCES:
        raise SystemExit(f"ABORT: unknown benchmark {benchmark!r}")
    if screen not in ("strict", "option_set"):
        raise SystemExit(f"ABORT: unknown screen {screen!r}")
    src = SOURCES[benchmark]
    if rows is None:
        rows = _load_hf(benchmark, split)
    n_upstream = len(rows)
    drops: dict = {}
    need_support = (require_support == "yes"
                    or (require_support == "auto" and benchmark == "sciq"))
    if benchmark == "obqa":
        raw = build_obqa(rows, split, drops)
        evidence_source = src["evidence_source"]
    elif benchmark == "sciq":
        raw = build_sciq(rows, split, drops, need_support)
        evidence_source = src["evidence_source"]
    else:
        raw = build_arc(rows, split, drops, arc_evidence, arc_corpus)
        evidence_source = src["evidence_source"] + "_STUB"
    if limit is not None:
        raw = raw[:limit]
    eps = [to_episode(r, benchmark, split, evidence_source, screen)
           for r in raw]
    stats = {
        "benchmark": benchmark, "split": split, "hf": src["hf"],
        "hf_config": src["config"], "evidence_source": evidence_source,
        "n_upstream": n_upstream, "n_built": len(eps), "limit": limit,
        "dropped": drops, "screen": screen,
        "require_support": need_support,
        "answer_position": assert_choice_balance(eps),
        "leakage": leakage_summary(eps),
    }
    return eps, stats


def leakage_summary(eps: list) -> dict:
    """Aggregate the §1.2 screens. Ratios are summarised, never thresholded."""
    n = max(len(eps), 1)
    ov = sorted(e["leakage_flags"]["evidence_question_overlap"] for e in eps)

    def q(p):
        return round(ov[min(int(p * (len(ov) - 1)), len(ov) - 1)], 4) if ov else 0.0

    return {
        "n": len(eps),
        "n_gold_verbatim": sum(e["leakage_flags"]["gold_verbatim"] for e in eps),
        "n_distractor_verbatim": sum(bool(e["leakage_flags"]["distractors_verbatim"])
                                     for e in eps),
        "n_answer_letter_pattern": sum(e["leakage_flags"]["answer_letter_pattern"]
                                       for e in eps),
        "n_evidence_empty": sum(e["leakage_flags"]["evidence_empty"] for e in eps),
        "n_violation_strict": sum(e["leakage_flags"]["violation_strict"] for e in eps),
        "n_violation_option_set": sum(e["leakage_flags"]["violation_option_set"]
                                      for e in eps),
        "frac_violation_strict": round(
            sum(e["leakage_flags"]["violation_strict"] for e in eps) / n, 4),
        "frac_violation_option_set": round(
            sum(e["leakage_flags"]["violation_option_set"] for e in eps) / n, 4),
        "overlap_ratio_quantiles": {"p10": q(0.10), "p50": q(0.50),
                                    "p90": q(0.90), "max": q(1.0)},
    }


def write_jsonl(path: str, eps: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for ep in eps:
            f.write(json.dumps(ep, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build a benchmark episode JSONL (BENCH_PLAN §1.1/§1.2).")
    ap.add_argument("--benchmark", required=True, choices=sorted(SOURCES))
    ap.add_argument("--split", required=True,
                    help="upstream split name (train | validation | test)")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N built items (smoke builds only)")
    ap.add_argument("--screen", default="strict", choices=["strict", "option_set"],
                    help="which §1.2 definition sets leakage_flags.violation: "
                         "strict = any option text verbatim or an answer letter; "
                         "option_set = a DISTRACTOR verbatim or an answer letter "
                         "(gold-in-evidence is the intended asymmetry for SciQ)")
    ap.add_argument("--drop-violations", action="store_true",
                    help="drop flagged items instead of only logging them; the "
                         "08-07 dev lock decides whether this is used")
    ap.add_argument("--require-support", default="auto",
                    choices=["auto", "yes", "no"],
                    help="auto = yes for sciq (884/1000 test items carry one), "
                         "no elsewhere")
    ap.add_argument("--arc-evidence", default="stub", choices=["stub", "bm25"])
    ap.add_argument("--arc-corpus", default=None,
                    help="ARC Corpus path for --arc-evidence bm25 (not implemented in B0)")
    ap.add_argument("--allow-stub-evidence", action="store_true",
                    help="required to WRITE an ARC file whose sender evidence is "
                         "a stub: such a file must never reach a packet arm")
    ap.add_argument("--allow-test", action="store_true",
                    help="acknowledge that a test split is the one-touch "
                         "confirmatory set (Hard Rule 6, §1.2)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    eps, stats = build_episodes(
        a.benchmark, a.split, limit=a.limit, screen=a.screen,
        require_support=a.require_support, arc_evidence=a.arc_evidence,
        arc_corpus=a.arc_corpus)

    if a.benchmark == "arc-c" and a.arc_evidence == "stub" and not a.allow_stub_evidence:
        raise SystemExit(
            "ABORT: ARC sender evidence is a STUB (empty) until the ARC Corpus "
            "+ BM25 retrieval land (BENCH_PLAN §5 item 1). Pass "
            "--allow-stub-evidence to build the receiver-side fields anyway; "
            "the resulting file is valid for the no_comm arm ONLY and must not "
            "be used for skyline/packet arms.")
    if "test" in a.split and not a.allow_test:
        raise SystemExit(
            f"ABORT: split {a.split!r} is the one-touch confirmatory set "
            "(Hard Rule 6; BENCH_PLAN §1.2 'test [non-English note removed] confirmatory "
            "launch [non-English note removed]'). Building the file is allowed under --allow-test; "
            "selection and dev calibration must use validation/train.")

    n_before = len(eps)
    if a.drop_violations:
        eps = [e for e in eps if not e["leakage_flags"]["violation"]]
    stats["n_dropped_violations"] = n_before - len(eps)
    stats["n_written"] = len(eps)
    if not eps:
        raise SystemExit(
            "ABORT instead of writing an empty file: zero episodes survived "
            "(run_j2 2026-08-03 lesson -- a DONE marker without a row count is "
            "not completion). Check --split/--limit/--require-support.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    write_jsonl(a.out, eps)
    stats.update({
        "out": os.path.abspath(a.out), "created_utc": utcnow(),
        "jsonl_sha256": sha256_file(a.out),
        "builder_sha256": sha256_file(os.path.abspath(__file__)),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "one_touch_test_split": "test" in a.split,
        "plan": "the study protocol §1.1/§1.2 + PI AMENDMENT 2026-08-05",
    })
    sum_path = a.out.replace(".jsonl", "") + ".summary.json"
    with open(sum_path, "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)

    lk = stats["leakage"]
    print(f"[bench] {a.benchmark}/{a.split}: upstream={stats['n_upstream']} "
          f"built={stats['n_built']} written={stats['n_written']} "
          f"dropped={stats['dropped']} violations_dropped={stats['n_dropped_violations']}",
          flush=True)
    print(f"[bench] leakage (§1.2): gold_verbatim={lk['n_gold_verbatim']} "
          f"distractor_verbatim={lk['n_distractor_verbatim']} "
          f"answer_letter={lk['n_answer_letter_pattern']} "
          f"violation_{a.screen}={lk['n_violation_' + a.screen]} "
          f"overlap_ratio p50={lk['overlap_ratio_quantiles']['p50']} "
          f"p90={lk['overlap_ratio_quantiles']['p90']} (LOGGED, not thresholded)",
          flush=True)
    print(f"[bench] answer positions {stats['answer_position']['fractions']} "
          f"(enforced={stats['answer_position']['enforced']})", flush=True)
    print(f"[bench] ALL DONE rows={stats['n_written']} -> {a.out} "
          f"(summary {sum_path})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
