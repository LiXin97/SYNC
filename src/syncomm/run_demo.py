"""J7 — THE ICONIC INTERSECTION DEMO (the study protocol §4).

One episode family that closes the whole conjunction the audit says no single
experiment closes (F1/F28, ADR-0006 item 2):

    T2 episode: two private evidence sentences, two aliases (GLAV / LORP)
      -> SENDER (family A) solves ITS OWN dictionary on EACH evidence
         sentence separately            [sender-derived extraction]
      -> ONE Packet with TWO Lanes: Lane(slot_id=0,...), Lane(slot_id=1,...)
      -> serialize -> bytes -> deserialize (checksum)   [only bytes cross]
      -> RECEIVER (family B != A) maps slot_id -> its own alias carrier span
         and writes with ITS OWN dictionary / gamma / frozen J0-C geometry
                                        [cross-family receiver, no pairwise map]
      -> forced-choice alias logit gap AND free greedy generation
                                        [slot sensitivity + Hard Rule 3]

The primary estimand is the sender-dependent, slot-aware, placebo-matched
interaction (§4.3 Q1)

    I_sender = (MC - MS) - (PC - PS)

MC/MS are the episode's OWN sender-extracted packet written into the correct /
swapped carrier slots; PC/PS are a DONOR episode's complete two-lane packet
(role-balanced derangement, matched on k / type histogram / bytes / q multiset /
realised ||delta||) written into the correct / swapped slots.  MC-MS alone is
slot sensitivity; PC-PS is the same slot manipulation with content that cannot
be about this episode, so the difference of differences is the part of the slot
effect that requires THIS sender's message.

Arms (the §4 nine, plus two opt-in ceilings):
  none          no write, neutral carriers
  text_oracle   carriers name the entities (full-information ceiling)
  MC            matched content, correct slots        <- own packet
  MS            matched content, swapped slots        <- own packet
  PC            donor content, correct slots          <- donor packet
  PS            donor content, swapped slots          <- donor packet
  text_packet   the SAME round-tripped packet rendered as text (co-primary)
  rand_dict     MC content, correct slots, decoded with the RANDOM dictionary
  rand_dict_swapped  the same, in SWAPPED slots (opt-in, ADDRPERM0908); the pair
                gives I_rand = (RC-RS) - (PC-PS), commensurable with I_sender, so
                "any sparse perturbation would do" becomes a measured contrast
                rather than evidence transported from another bank
  zero_packet   all-zero packet reconstructed and added -> exact no-op
  HC / HS       (opt-in) oracle canonical single-ID addresses, correct/swapped;
                gives I_oracle = (MC-MS) - (HC-HS), commensurable with J2a

Defects this runner fixes by construction (audit §1, §2.1):
  F16  every packet is serialize()d and deserialize()d; only the round-tripped
       object is consumed, by BOTH the injection and the text path.
  F17  row metadata is derived from the packet ACTUALLY used (after donor
       substitution), and asserted against it on every row.
  N1   donors that would imply the recipient's own answer are inadmissible.
  N2   role-balanced derangement: every packet appears exactly once in the gold
       role and exactly once in the wrong role, so packet identity is
       orthogonal to role and the role contrast is identified.
  F24  realised ||delta||_2 per layer is measured and logged on every row, so
       the falsifier panel can be amplitude-matched post hoc.
  ADDENDUM2 (run_j2)  the alias->entity mapping comes from slots{}, never from
       provenance.entity_words (which inverts on every twin).
  HARD_RULE_3_GAP  both endpoints: forced choice AND free greedy generation
       (unanswered = incorrect, coverage recorded).

Falsifier panel (§4.4): --dict-variant re-runs the same episodes, the same
donor assignment and the same arms with any dictionary from
artifacts/jlens2/{model}_{variant}_inv2.pt; rows carry dict_variant and the
realised ||delta|| so cells can be amplitude-matched post hoc (or matched a
priori with --amp-override).

  python3 -m syncomm.run_demo --sender qwen3-8b-base --receiver llama-3.1-8b \
      --bank data/inventory_v2/t2_dev.jsonl \
      --inventory data/inventory_v2/inventory.json \
      --facts data/inventory_v2/facts_attributes.json \
      --n-clusters 200 --reps 2 --out artifacts/j7

Resumable (key = episode|arm|dict_variant), deterministic (fixed seeds; the
donor assignment is frozen to assignment.json before scoring and reused on
resume).
"""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import os
import re
import sys
import zlib

import torch

from .e015.core import load, tap_layer, append_jsonl, done_keys
from .packet import (Lane, Packet, deserialize, dequantize, inventory_digest,
                     rate_bytes, reconstruct, render_text, serialize)
from .run_j0c import mean_h, LITERAL
from .run_j2 import CARRIER, ORACLE, FEWSHOT_T2, first_tok, sent_span
from .run_j3 import GAMMA, BAND_TAPS
from .writer import SlotWrite, _solve_a0, USE_GPU_SOLVER
from . import solver as _solver
from .cost_timing import (Timer as _CostTimer, time_text_gen as _time_text_gen,   # cost axis (2026-09-12)
                          summarize as _cost_summarize, environment as _cost_env)

# ---------------------------------------------------------------- constants

ARMS = ("none", "text_oracle", "MC", "MS", "PC", "PS",
        "text_packet", "rand_dict", "zero_packet")
# 2026-09-13: rand_dict_normfix writes the RANDOM dictionary's direction rescaled, site by site,
# to the norm the REAL dictionary's write would have had on that episode -- the amplitude control
# the external review showed row normalization does not give.
OPTIONAL_ARMS = ("HC", "HS", "rand_dict_swapped",
                 "rand_dict_normfix", "rand_dict_normfix_swapped")
WRITE_ARMS = ("MC", "MS", "PC", "PS", "rand_dict", "rand_dict_swapped",
              "rand_dict_normfix", "rand_dict_normfix_swapped",
              "zero_packet", "HC", "HS")
RAND_ARMS = ("rand_dict", "rand_dict_swapped",
             "rand_dict_normfix", "rand_dict_normfix_swapped")   # all decode through RND
NORMFIX_ARMS = ("rand_dict_normfix", "rand_dict_normfix_swapped")
NORMFIX_REFERENCE = {"rand_dict_normfix": "MC", "rand_dict_normfix_swapped": "MS"}
DONOR_ARMS = ("PC", "PS")
SWAPPED_ARMS = ("MS", "PS", "HS", "rand_dict_swapped", "rand_dict_normfix_swapped")
ROLE = {"none": "none", "text_oracle": "oracle_text", "MC": "gold",
        "MS": "gold_swapped", "PC": "donor", "PS": "donor_swapped",
        "text_packet": "text", "rand_dict": "random_dict",
        "rand_dict_swapped": "random_dict_swapped",
        "rand_dict_normfix": "random_dict_norm_matched",
        "rand_dict_normfix_swapped": "random_dict_norm_matched_swapped",
        "zero_packet": "zero", "HC": "oracle", "HS": "oracle_swapped"}

TOPK = 4
QBITS = 8
LAM1, LAM2 = 0.05, 1e-3
BALANCE_LO, BALANCE_HI = 0.45, 0.55

# Free-generation few-shot: the SAME two demonstrations as the forced-choice
# few-shot with the answer options removed from the question, so the episode's
# own question never lists its options (Hard Rule 3 / refocus 02 §13) while the
# answer TYPE (an alias) is still demonstrated.
FEWSHOT_T2_FREE = ("ZEPH is a violin. QUOR is a pigeon.\n"
                   "Q: Which entity can fly?\nA: QUOR\n\n"
                   "TRAB is a hoe. MYLK is a stool.\n"
                   "Q: Which entity is sat on?\nA: MYLK\n\n")

# P4 (gates/confirmatory_20260818.lock.DRAFT.json): free-generation templates
# 1 and 2, frozen VERBATIM in the preregistration note (2026-08-12;
# sha pinned in the lock). Carrier sentences stay bit-identical across
# templates -- the write spans anchor on them; templates vary ONLY the
# few-shot header and the question/answer framing. Template 0 is
# FEWSHOT_T2_FREE above (identity, the dev-era prompt).
FEWSHOT_T2_FREE_T1 = ("DRIF is a kettle. SNUB is a ladder.\n"
                      "Q: Which entity is climbed?\nA: SNUB\n\n"
                      "PLOM is a trumpet. VASK is a carpet.\n"
                      "Q: Which entity is walked on?\nA: VASK\n\n")
FEWSHOT_T2_FREE_T2 = ("GRELL is a candle. THIMP is a bicycle.\n"
                      "Question: Which entity has wheels?\nAnswer: THIMP\n\n"
                      "SPROD is a mirror. KLUN is a blanket.\n"
                      "Question: Which entity shows a reflection?\n"
                      "Answer: SPROD\n\n")

# Model -> family. The demo is cross-family by contract; a same-family link
# needs --allow-same-family and is labelled as an instrument, not a result.
FAMILY = {"qwen3-8b-base": "qwen", "qwen3-14b-base": "qwen",
          "qwen3-32b": "qwen", "qwen3-8b": "qwen",
          "llama-3.1-8b": "llama", "llama-3.1-8b-instruct": "llama",
          "phi-4": "phi", "glm-4-9b": "glm", "granite-4.1-8b": "granite",
          "mistral-nemo": "mistral", "yi-1.5-9b-chat": "yi",
          "gpt-oss-20b": "gptoss", "qwen2.5-7b": "qwen2.5"}


class SenderRequired(RuntimeError):
    """Raised when packet construction is attempted without a sender solve.

    Requirement 1 of the demo: packets are sender-derived BY CONSTRUCTION.
    There is no code path from the gold entity id to a communicated packet.
    """


def family(model: str) -> str:
    return FAMILY.get(model, model)


# ------------------------------------------------------- episode selection

def _stride(n: int) -> int:
    """Largest odd step coprime with n, so consecutive picks cycle all combos."""
    for s in range(n // 2 + 1, 0, -1):
        if _gcd(s, n) == 1:
            return s
    return 1


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def select_episodes(rows, n_clusters: int, reps: int = 2,
                    cluster_offset: int = 0):
    """Balanced twin/order/paraphrase sampling (run_j2 ADDENDUM2, generalised).

    Cluster i contributes `reps` realisations, picked by walking the cluster's
    (twin, paraphrase, order) combinations with a coprime stride so that (a)
    with reps=2 every cluster contributes exactly one twin-0 and one twin-1
    episode, and (b) with reps=1 twins alternate across clusters. Taking index 0
    of each cluster instead made the gold answer the FIRST alias in 199/199
    episodes and saturated the none arm (run_j2 ADDENDUM2).
    """
    by_cluster = collections.defaultdict(list)
    for r in rows:
        by_cluster[r["cluster_id"]].append(r)
    eps = []
    kept = 0
    for i, cid in enumerate(sorted(by_cluster)):
        if i < cluster_offset:
            continue
        combos = sorted(by_cluster[cid],
                        key=lambda r: (r["twin_index"], r["paraphrase_idx"],
                                       r["option_order"]))
        n = len(combos)
        st = _stride(n)
        for j in range(min(reps, n)):
            eps.append(combos[((i * reps + j) * st) % n])
        kept += 1
        if kept >= n_clusters:
            break
    return eps


def gold_first_fraction(eps) -> float:
    if not eps:
        return float("nan")
    hits = [e["forced_choice"]["gold"] == e["public_schema"]["aliases"]["entity_1"]
            for e in eps]
    return sum(hits) / len(hits)


def assert_balanced(eps, lo: float = BALANCE_LO, hi: float = BALANCE_HI) -> float:
    """The run_j2 ADDENDUM2 position guard: abort on a position confound."""
    frac = gold_first_fraction(eps)
    if not lo <= frac <= hi:
        raise SystemExit(f"ABORT: position confound, gold-is-first-alias "
                         f"{frac:.3f} outside [{lo},{hi}] over {len(eps)} "
                         f"episodes (run_j2 ADDENDUM2)")
    return frac


# ------------------------------------------------- slots-derived alias map

def alias_entity_map(ep, addr2word):
    """alias slot -> entity word, from slots{} ONLY.

    run_j2 ADDENDUM2 fix 2: provenance.entity_words is in fixed
    [positive, negative] order and INVERTS the mapping on every twin-1 episode
    (pilot: oracle 1.000 on twin0 vs 0.033 on twin1). slots{} is the per-twin
    ground truth and is the only admissible source.
    """
    out = {}
    for slot in ("entity_1", "entity_2"):
        addr = ep["slots"].get(slot)
        out[slot] = addr2word.get(addr)
    return out


def episode_entities(ep, addr2word):
    m = alias_entity_map(ep, addr2word)
    return (m["entity_1"], m["entity_2"])


# ------------------------------------------------------- packet construction

def quantize_topk(a: torch.Tensor, topk: int = TOPK, qbits: int = QBITS,
                  floor: float = 1e-6):
    """Coefficient vector -> (ascending support ids, q multiset).

    Sender-side max-normalisation (run_j3.py:111 convention): the largest
    coefficient is pinned to the top code, so absolute magnitude never crosses
    the wire (J1 weight-causality FAIL; audit F9).
    """
    k = min(topk, a.numel())
    top = torch.topk(a, k)
    S = sorted(int(i) for i in top.indices if float(a[i]) > floor)
    if not S:
        return (), ()
    vals = a[S]
    qmax = (1 << qbits) - 1 if qbits else 0
    if qbits == 0:
        return tuple(S), tuple(0 for _ in S)
    q = tuple(int(round(qmax * float(v / vals.max()))) for v in vals)
    return tuple(S), q


def build_packet(sender_solve, evidence, digest: str, *, topk: int = TOPK,
                 qbits: int = QBITS):
    """ONE two-lane packet from the SENDER's solve on the lane evidence.

    `sender_solve` maps an evidence STRING to a coefficient vector over the
    canonical concept ids. It is the only source of packet content: this
    function never sees slots{}, entity ids or the gold answer, so a packet is
    not constructible without the sender model (requirement 1).
    """
    if sender_solve is None:
        raise SenderRequired(
            "no sender solve: packets in this demo are sender-derived by "
            "construction (audit F12 — a sender-free packet makes the whole "
            "estimand vacuous). Load the sender model or do not run.")
    lanes = []
    for slot_id, ev in enumerate(evidence):
        a = sender_solve(ev)
        S, q = quantize_topk(a, topk=topk, qbits=qbits)
        if not S:
            return None
        lanes.append(Lane(slot_id, S, q))
    return Packet(digest, qbits, tuple(lanes))


def build_sender_packets(eps, sender_solve, digest, *, topk=TOPK, qbits=QBITS):
    """episode_id -> round-tripped Packet. Fails loudly without a sender."""
    if sender_solve is None:
        raise SenderRequired(
            "build_sender_packets called with sender_solve=None: the demo has "
            "no packet source other than the sender model.")
    out, dropped = {}, []
    for ep in eps:
        ev = (ep["sender_evidence"]["entity_1"], ep["sender_evidence"]["entity_2"])
        pkt = build_packet(sender_solve, ev, digest, topk=topk, qbits=qbits)
        if pkt is None:
            dropped.append(ep["episode_id"])
            continue
        out[ep["episode_id"]] = round_trip(pkt)
    return out, dropped


def round_trip(pkt: Packet) -> Packet:
    """serialize -> bytes -> deserialize, with the checksum actually exercised.

    Audit F16: the shipped J3 run asserted "only a serialized packet crosses
    the boundary" and never called deserialize. Here the round-tripped object
    is the ONLY one any downstream path may consume.
    """
    wire = serialize(pkt)
    if len(wire) != rate_bytes(pkt):
        raise ValueError(f"wire length {len(wire)} != rate_bytes {rate_bytes(pkt)}")
    rx = deserialize(wire)              # raises ValueError on checksum failure
    if rx != pkt:
        raise ValueError("packet changed across the wire round trip")
    return rx


def packet_hex(pkt: Packet) -> str:
    return serialize(pkt).hex()


def packet_sha256(pkt: Packet) -> str:
    return hashlib.sha256(serialize(pkt)).hexdigest()[:16]


# --------------------------------------------------------- donor matching

def type_hist(ids, types):
    return tuple(sorted(collections.Counter(types[i] for i in ids).items()))


def lane_signature(lane: Lane, types):
    """The hard shape of a lane: support size + concept-type histogram."""
    return (len(lane.concept_ids), type_hist(lane.concept_ids, types))


def packet_signature(pkt: Packet, types):
    """§3.4 clauses 1, 2, 4 collapsed into one comparable key."""
    return (tuple(lane_signature(ln, types) for ln in pkt.lanes), rate_bytes(pkt))


def q_max_diff(pkt: Packet, other: Packet) -> int:
    """§3.4 clause 3: max |q_(i) - q'_(i)| over descending-sorted q, per lane."""
    worst = 0
    for la, lb in zip(pkt.lanes, other.lanes):
        qa = sorted(la.q, reverse=True)
        qb = sorted(lb.q, reverse=True)
        n = min(len(qa), len(qb))
        for i in range(n):
            worst = max(worst, abs(qa[i] - qb[i]))
        for i in range(n, max(len(qa), len(qb))):
            worst = max(worst, qa[i] if i < len(qa) else qb[i])
    return worst


def implies_same_answer(rec_ents, don_ents, attribute, facts) -> bool:
    """§3.4 clause 6 / defect N1: a donor whose entities decide the recipient's
    question the same way is inadmissible (that is 'wrong content' containing
    the right content). Pattern = (lane0 has the attribute, lane1 has it)."""
    if not facts or attribute is None:
        return False

    def pat(ents):
        return tuple(bool(facts.get(e, {}).get(attribute) is True) for e in ents)

    rp, dp = pat(rec_ents), pat(don_ents)
    return rp == dp and dp[0] != dp[1]


def donor_admissible(rec_pkt, don_pkt, *, types, rec_ents, don_ents,
                     attribute=None, facts=None, rec_ent_ids=None):
    """§3.4 hard clauses. Returns (ok, reasons) — reasons is empty iff ok.

    `rec_ent_ids` (optional) are the recipient's own entity concept ids: a
    donor whose SUPPORT names one of them is not wrong content, whatever
    entity it was extracted from (defect N1 at the support level).
    """
    reasons = []
    if rec_ent_ids:
        ids = {i for ln in don_pkt.lanes for i in ln.concept_ids}
        if ids & {i for i in rec_ent_ids if i is not None}:
            reasons.append("support_names_recipient")
    if packet_signature(rec_pkt, types) != packet_signature(don_pkt, types):
        sig_r = packet_signature(rec_pkt, types)
        sig_d = packet_signature(don_pkt, types)
        if [l[0] for l in sig_r[0]] != [l[0] for l in sig_d[0]]:
            reasons.append("k")
        if [l[1] for l in sig_r[0]] != [l[1] for l in sig_d[0]]:
            reasons.append("type_hist")
        if sig_r[1] != sig_d[1]:
            reasons.append("bytes")
    if set(rec_ents) & set(don_ents):
        reasons.append("entity_overlap")
    if implies_same_answer(rec_ents, don_ents, attribute, facts):
        reasons.append("implies_same_answer")
    return (not reasons), reasons


def delta_rel_diff(self_norm: float, donor_norm: float) -> float:
    """§3.4 clause 5: | ||delta'|| - ||delta|| | / ||delta||."""
    if self_norm <= 0:
        return float("inf")
    return abs(donor_norm - self_norm) / self_norm


def donor_cost(qd: int, rd: float, q_tol: int, delta_tol: float) -> float:
    """Soft clauses 3 and 5: zero inside tolerance, then argmin (§3.4)."""
    return (max(0.0, (qd - q_tol) / 255.0) + max(0.0, rd - delta_tol)
            + 1e-3 * (qd / 255.0) + 1e-3 * min(rd, 1e6))


# ------------------------------------------------ role-balanced derangement

def _augment(r, feasible, taken, seen):
    for d in feasible.get(r, ()):
        if d in seen:
            continue
        seen.add(d)
        if d not in taken or _augment(taken[d], feasible, taken, seen):
            taken[d] = r
            return True
    return False


def _max_matching(keys, feasible):
    taken = {}                                  # donor -> recipient
    for r in keys:
        _augment(r, feasible, taken, set())
    return {rec: don for don, rec in taken.items()}


def role_balanced_derangement(keys, feasible, cost=None, *, rounds: int = 3,
                              seed: int = 0, scan: int = 64):
    """A bijection sigma on a subset S of `keys` with sigma(i) != i and
    feasible(i, sigma(i)) for all i — every packet appears EXACTLY ONCE in the
    gold role (its own episode) and EXACTLY ONCE in the wrong role (§3.4
    clause 7). This is the fix for defect N2: with role balanced this way, a
    packet fixed effect is no longer collinear with the role indicator.

    `feasible` maps key -> iterable of admissible donors (self excluded by this
    function anyway). `cost` (optional) is cost(recipient, donor) -> float,
    minimised by bounded 2-opt swaps that preserve feasibility (the soft
    clauses 3 and 5, which have an explicit argmin fallback in §3.4).

    Returns (assignment, dropped). Keys that cannot be part of any such
    bijection are dropped rather than silently self-matched.
    """
    S = [k for k in keys]
    feas = {k: [d for d in feasible.get(k, ()) if d != k] for k in S}
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10 * len(S) + 1000))
    try:
        while True:
            live = set(S)
            f = {k: [d for d in feas[k] if d in live] for k in S}
            m = _max_matching(S, f)
            if len(m) == len(S):
                break
            S = [k for k in S if k in m]
            if not S:
                return {}, list(keys)
    finally:
        sys.setrecursionlimit(limit)
    dropped = [k for k in keys if k not in m]
    if cost is None:
        return m, dropped
    # 2-opt on the soft clauses; deterministic order, feasibility preserved,
    # bounded candidate scan (cost() touches the receiver's residual geometry).
    live = set(m)
    fl = {k: set(d for d in feas[k] if d in live) for k in m}
    cur = {k: cost(k, m[k]) for k in m}
    order = sorted(m)
    rng = torch.Generator().manual_seed(seed)
    for _ in range(rounds):
        moved = 0
        hot = sorted(order, key=lambda k: (-cur[k], k))
        perm = torch.randperm(len(order), generator=rng).tolist()
        for i in hot:
            if cur[i] <= 0.0:
                continue
            tried = 0
            for pi in perm:
                j = order[pi]
                if j == i:
                    continue
                di, dj = m[i], m[j]
                if dj == i or di == j:
                    continue
                if dj not in fl[i] or di not in fl[j]:
                    continue
                tried += 1
                if tried > scan:
                    break
                ci, cj = cost(i, dj), cost(j, di)
                if ci + cj < cur[i] + cur[j] - 1e-12:
                    m[i], m[j] = dj, di
                    cur[i], cur[j] = ci, cj
                    moved += 1
                    break
        if not moved:
            break
    return m, dropped


# ------------------------------------------------------- free generation

def strip_options(question: str) -> str:
    """'Which entity is worn? Answer with GLAV or LORP.' -> 'Which entity is worn?'"""
    return re.sub(r"\s*Answer with\b[^?]*?\.\s*$", "", question.strip()).strip()


def _frame_qa(question: str) -> str:
    """Templates 0/1 framing `Q: {question}\\nA:` (prereg). The options are
    always stripped -- the episode's own question never lists them (Hard Rule
    3, a frozen invariant across ALL templates) -- so template 0 renders the
    dev-era prompt tail byte-for-byte."""
    return f"Q: {strip_options(question)}\nA:"


def _frame_question_answer(question: str) -> str:
    """Template 2 framing `Question: {strip_options(question)}\\nAnswer:`."""
    return f"Question: {strip_options(question)}\nAnswer:"


# index -> (few-shot header, question-framing fn); order is the prereg's 0/1/2
# (the preregistration note). Template 1 rewrites the few-shot
# content only; template 2 rewrites the framing too.
FREE_TEMPLATES = ((FEWSHOT_T2_FREE, _frame_qa),
                  (FEWSHOT_T2_FREE_T1, _frame_qa),
                  (FEWSHOT_T2_FREE_T2, _frame_question_answer))


def free_prompt(head: str, question: str, template: int = 0,
                answer_prefix: str = "") -> str:
    """The full free-generation prompt for one frozen P4 template. `head` is
    the carrier/oracle head (+ optional packet text) and is bit-identical
    across templates BY CONSTRUCTION: a template chooses only the few-shot
    header and the question framing, never the carriers the writes anchor on.

    `answer_prefix` (FREEGEN0906, 2026-09-05; default "" = byte-identical to
    every earlier run) is appended AFTER the frame's answer cue, e.g. the empty
    thinking block '<think>\n\n</think>\n\n' that lets a thinking-mode
    receiver answer inside a short budget. It touches only the free-generation
    prompt; the forced-choice prompt, carriers and write positions are untouched,
    which the FREEGEN0906 replay check verifies row by row."""
    header, frame = FREE_TEMPLATES[template]
    return header + head + frame(question) + answer_prefix


def free_key(key: str, template_idx: int, rollout_idx=None) -> str:
    """Resume key for one ADDITIONAL P4 free-gen unit. The main arm row keeps
    the dev-era key (episode|arm|dict_variant) and carries the template-0
    greedy generation; every other (template, rollout) unit appends a suffix
    so done_keys() skips completed units independently on resume
    (rollout_idx None = greedy)."""
    r = "greedy" if rollout_idx is None else f"r{rollout_idx}"
    return f"{key}|freegen|t{template_idx}|{r}"


def derive_rollout_seed(base_seed: int, episode_id: str, arm: str,
                        template_idx: int) -> int:
    """P4 sensitivity seed, frozen formula (prereg + lock P4):
    seed_r = base_seed_r XOR crc32(f"{episode_id}|{arm}|{template_idx}")."""
    return base_seed ^ zlib.crc32(f"{episode_id}|{arm}|{template_idx}".encode())


def score_free(text: str, gold: str, cf: str):
    """(answered, correct) on free greedy generation.

    Whichever alias is named FIRST is the answer; naming neither is UNANSWERED
    and scored INCORRECT (never answered-only — HARD_RULE_3_GAP_2026-08-02,
    refocus 02 §13). Coverage is reported separately from accuracy.
    """
    t = (text or "").upper()
    ig = _first_word_index(t, gold.upper())
    ic = _first_word_index(t, cf.upper())
    if ig is None and ic is None:
        return False, False
    if ig is None:
        return True, False
    if ic is None:
        return True, True
    return True, ig < ic


def _first_word_index(text: str, word: str):
    m = re.search(r"(?<![A-Z0-9])" + re.escape(word) + r"(?![A-Z0-9])", text)
    return m.start() if m else None


# ------------------------------------------------------------- dictionaries

PRIMARY_VARIANTS = ("jlens2_centered", "centered", "tap0.5", "primary")


def dict_path(dict_dir: str, model: str, variant: str) -> str:
    """artifacts/jlens2/{model}_{variant}_inv2.pt; the primary dictionary keeps
    its shipped filename {model}_tap0.5_inv2.pt (§4.4 panel row 1)."""
    tag = "tap0.5" if variant in PRIMARY_VARIANTS else variant
    return os.path.join(dict_dir, f"{model}_{tag}_inv2.pt")


def load_dict(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    D, words = blob["dirs"].float(), blob["words"]
    sha = hashlib.sha256(D.contiguous().numpy().tobytes()).hexdigest()[:16]
    return D, words, sha


def model_devices(model) -> list[str]:
    """Sorted distinct parameter devices of a loaded model.

    Recorded in the run summary and the log so that a device-map equivalence
    control is CAPABLE OF FAILING (ADR-p3stage2-sender-devicemap-20260901: a
    'sharded' arm whose model actually sits on one device is VOID, the P4
    trap). Same statistic p3roster_acc1 / e015.run_pilot record.
    """
    return sorted({str(p.device) for p in model.parameters()})


# --------------------------------------------------------- write machinery

def build_plan(pkt: Packet, swapped: bool, mass_fn):
    """span index -> (support, a_hat).

    Lane slot_id s addresses carrier span s; `swapped` exchanges the two spans
    while transmitting byte-identical content (arm MS/PS: perfectly matched on
    k, q, types and bytes BY CONSTRUCTION — the realised ||delta|| is measured
    per row rather than assumed).
    """
    plan = {}
    for ln in pkt.lanes:
        span = (1 - ln.slot_id) if swapped else ln.slot_id
        plan[span] = (list(ln.concept_ids), mass_fn(ln))
    return plan


def mass_scaler(qbits: int, mass: float):
    """Receiver-side calibration: dequantized coefficients rescaled to the
    receiver's OWN L1 mass gamma_r * m_r (run_j3.scaled; pairwise-free)."""
    def fn(ln: Lane):
        v = dequantize(ln.q, qbits)
        s = float(v.sum())
        return v * (mass / s) if s > 1e-8 else v
    return fn


@torch.no_grad()
def measure_sites(model, tok, prompt, band, span_positions, D, dev):
    """Pre-write coefficients a0 and ||h|| at every write site, from ONE
    unwritten forward. Returns (sites, A0 [M, C], hnorm [M]).

    a0 is the same quantity writer.SlotWrite solves inside the hook (same
    entry point `_solve_a0`, same lam1/lam2). It is exact for the lowest hooked
    layer; for the upper band layers it is measured on the UNPERTURBED forward
    (the hook would see the lower layer's write), so the logged ||delta|| is an
    unbiased matched quantity across arms rather than a bitwise replay. It is
    identical in construction for the matched and the donor packet, which is
    what §3.4 clause 5 needs.
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states
    sites, H = [], []
    for li in band:
        h = hs[li + 1][0]
        for span, poss in enumerate(span_positions):
            for p in poss:
                if 0 <= p < h.shape[0]:
                    sites.append((li, span, p))
                    H.append(h[p].float())
    if not sites:
        return [], None, None
    H = torch.stack(H)
    Dw = D.to(H.device) if USE_GPU_SOLVER else D.cpu()
    A0 = _solve_a0(H.to(Dw.device), Dw, LAM1, LAM2)
    return sites, A0.float(), H.norm(dim=-1).float().cpu()


def delta_stats(sites, A0, hnorm, D, plan):
    """Realised write norms: ||delta_site||_2 = ||(a_hat - a0[S]) D[S]||_2.

    Returns (mean ||delta||, per-layer mean, mean ||h||, delta/h). Logged on
    every row so the §4.4 panel can be amplitude-matched post hoc (audit F24:
    the shipped J4 rows recorded neither a0 nor ||delta||, which made the
    residual dose mismatch unquantifiable)."""
    if not sites or A0 is None:
        return 0.0, {}, 0.0, 0.0
    dev = A0.device
    rows_by_span = collections.defaultdict(list)
    for m, (li, span, _p) in enumerate(sites):
        rows_by_span[span].append((m, li))
    per_layer = collections.defaultdict(list)
    norms = []
    for span, rows in sorted(rows_by_span.items()):
        if span not in plan:
            continue
        S, ah = plan[span]
        if not S:
            continue
        idx = torch.tensor(S, dtype=torch.long, device=dev)
        ms = torch.tensor([m for m, _ in rows], dtype=torch.long, device=dev)
        # ONE matmul per span: [M_s, k] @ [k, d]
        resid = (ah.float().to(dev).unsqueeze(0)
                 - A0.index_select(0, ms).index_select(1, idx))
        nn = (resid @ D.index_select(0, idx)).norm(dim=-1).tolist()
        for (_m, li), n in zip(rows, nn):
            norms.append(n)
            per_layer[li].append(n)
    if not norms:
        return 0.0, {}, float(hnorm.mean()) if hnorm is not None else 0.0, 0.0
    mean_delta = sum(norms) / len(norms)
    mean_h = float(hnorm.mean()) if hnorm is not None else 0.0
    return (mean_delta,
            {str(li): sum(v) / len(v) for li, v in sorted(per_layer.items())},
            mean_h, (mean_delta / mean_h if mean_h > 0 else 0.0))


def site_norm_map(sites, A0, D, plan):
    """{(layer, position): ||delta_site||} for one episode's plan.

    delta_stats returns means; the 2026-09-13 external review showed that matching a control on
    row norms does not match the realised write, so a control has to be matched SITE BY SITE.
    This is the same computation as delta_stats, kept separate so that function's output (logged
    on every row of every historical run) does not move."""
    out = {}
    if not sites or A0 is None:
        return out
    dev = A0.device
    by_span = collections.defaultdict(list)
    for m, (li, span, p) in enumerate(sites):
        by_span[span].append((m, li, p))
    for span, rows in sorted(by_span.items()):
        if span not in plan:
            continue
        S, ah = plan[span]
        if not S:
            continue
        idx = torch.tensor(S, dtype=torch.long, device=dev)
        ms = torch.tensor([m for m, _, _ in rows], dtype=torch.long, device=dev)
        resid = (ah.float().to(dev).unsqueeze(0)
                 - A0.index_select(0, ms).index_select(1, idx))
        nn = (resid @ D.index_select(0, idx)).norm(dim=-1).tolist()
        for (_m, li, p), n in zip(rows, nn):
            out[(int(li), int(p))] = float(n)
    return out


def norm_control_for_arm(arm, eid, plan, reference_norms, *, want_norms, record_all):
    """Choose the reference target and optional recorder independently.

    Recording every write must not bypass matching a norm-control arm to its
    MC/MS reference. Both may be active in the same forward.
    """
    target, recorder = None, None
    if arm in NORMFIX_ARMS and plan is not None:
        ref = NORMFIX_REFERENCE[arm]
        target = reference_norms.get((ref, eid))
        if not target:
            raise SystemExit(f"ABORT: {arm} has no recorded {ref} write for {eid}; {ref} "
                             "must run first in the same run or the control is unmatched")
    if ((want_norms and arm in NORMFIX_REFERENCE.values())
            or (record_all and plan is not None)):
        recorder = {}
    return target, recorder


@torch.no_grad()
def forced_gap(model, tok, prompt, gid, cid, dev, *, band=None, D=None,
               plan=None, span_positions=None, vec=None, target_norms=None,
               record_norms=None):
    """Forced-choice logit gap under an optional write. Returns (gap, extras)."""
    hooks = []
    try:
        if plan:
            for li in band:
                for span, (S, ah) in sorted(plan.items()):
                    pos = span_positions[span]
                    if not pos or not S:
                        raise SystemExit(
                            f"ABORT: empty write (span={span}, |S|={len(S)}) — "
                            "a silent no-op would be byte-identical to `none`")
                    _target_norm_here = (
                        None if target_norms is None else
                        {p: target_norms[(li, p)] for p in pos if (li, p) in target_norms})
                    hooks.append(SlotWrite(model, li, [pos], "replace", D=D,
                                           support=list(S), a_hat=ah,
                                           lam1=LAM1, lam2=LAM2,
                                           target_norm=_target_norm_here or None,
                                           record_norms=record_norms))
        elif vec is not None:
            for li in band:
                for span, pos in enumerate(span_positions):
                    if pos:
                        hooks.append(SlotWrite(model, li, [pos], "add",
                                               vec=vec[span]))
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with _CostTimer() as _t:
            logits = model(ids).logits[0, -1].float()
    finally:
        for h in hooks:
            h.remove()
    lp = torch.log_softmax(logits, dim=-1)
    top1 = int(logits.argmax())
    ent = float(-(lp.exp() * lp).sum())
    n_writes = sum(h.n_writes for h in hooks)
    return float(logits[gid] - logits[cid]), {
        "logp_gold": float(lp[gid]), "logp_cf": float(lp[cid]),
        "entropy": ent, "answerable": top1 in (gid, cid), "n_writes": n_writes,
        # cost axis (2026-09-12, additive): receiver prefill tokens / synchronised wall-clock
        # and the in-hook a0 solve time summed over the installed hooks
        "prompt_tokens": int(ids.shape[1]), "prefill_ms": round(_t.ms, 3),
        "hook_solve_ms": round(sum(h.t_solve_s for h in hooks) * 1000.0, 3),
        "n_rescaled": sum(getattr(h, "n_rescaled", 0) for h in hooks),
        "rescale_median": (lambda f: (sorted(f)[len(f) // 2] if f else None))(
            [x for h in hooks for x in getattr(h, "rescale_factors", [])]),
        # what was really written, in the forward that produced this gap: delta_norm on the row is
        # computed outside the forward and is a proxy above the lowest layer
        "delta_norm_applied": (lambda v: (sum(v) / len(v) if v else None))(
            [x for h in hooks for x in getattr(h, "applied_norms", [])])}


@torch.no_grad()
def free_generate(model, tok, prompt, dev, *, band=None, D=None, plan=None,
                  span_positions=None, vec=None, max_new_tokens=8,
                  temperature=None, gen_seed=None):
    """Greedy continuation under the SAME write (prefill-only hooks condition
    generation through ordinary KV attention). temperature/gen_seed (P4
    rollout sensitivity) switch ONLY the decode call to seeded sampling; the
    hook path is byte-identical to the greedy one."""
    hooks = []
    try:
        if plan:
            for li in band:
                for span, (S, ah) in sorted(plan.items()):
                    pos = span_positions[span]
                    if pos and S:
                        hooks.append(SlotWrite(model, li, [pos], "replace", D=D,
                                               support=list(S), a_hat=ah,
                                               lam1=LAM1, lam2=LAM2))
        elif vec is not None:
            for li in band:
                for span, pos in enumerate(span_positions):
                    if pos:
                        hooks.append(SlotWrite(model, li, [pos], "add",
                                               vec=vec[span]))
        enc = tok(prompt, return_tensors="pt")
        ids = enc.input_ids.to(dev)
        if temperature is None:
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        else:
            if gen_seed is not None:
                torch.manual_seed(gen_seed)
            out = model.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=temperature,
                                 pad_token_id=tok.pad_token_id)
    finally:
        for h in hooks:
            h.remove()
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


# ------------------------------------------------------------------- runner

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--facts", default=None,
                    help="facts_attributes.json — enables the N1 donor check "
                         "(a donor that implies the recipient's answer is "
                         "inadmissible). Strongly recommended.")
    ap.add_argument("--dict-dir", default="artifacts/jlens2")
    ap.add_argument("--dict-variant", default="jlens2_centered",
                    help="§4.4 falsifier panel: any {model}_{variant}_inv2.pt "
                         "(jlens2_centered|raw|unembed|common|random|probe|...)")
    ap.add_argument("--sender-dict-variant", default=None,
                    help="default: same as --dict-variant (the protocol runs "
                         "one dictionary family end to end)")
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--cluster-offset", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2,
                    help="realisations per cluster; 2 gives exact per-cluster "
                         "twin balance")
    ap.add_argument("--arms", default=None,
                    help=f"comma list; default {','.join(ARMS)} "
                         f"(opt-in: {','.join(OPTIONAL_ARMS)})")
    ap.add_argument("--gamma", type=float, default=None,
                    help="receiver dose; default = its frozen J0-C gamma")
    ap.add_argument("--amp-override", type=float, default=None,
                    help="absolute L1 write mass, skipping the per-dictionary "
                         "magnitude probe (a-priori amplitude matching across "
                         "the falsifier panel; required for rank-1 `common`)")
    ap.add_argument("--n-mag", type=int, default=40)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--qbits", type=int, default=QBITS, choices=[0, 2, 4, 8])
    ap.add_argument("--q-tol", type=int, default=26,
                    help="§3.4 clause 3 tolerance on the sorted q multiset")
    ap.add_argument("--delta-tol", type=float, default=0.05,
                    help="§3.4 clause 5 tolerance on realised ||delta||")
    ap.add_argument("--receiver-device-map", default=None,
                    help="'auto' shards the RECEIVER across visible GPUs. Default None keeps every existing campaign byte-identical; required only for receivers larger than one card (Qwen3.5-397B-A17B).")
    ap.add_argument("--sender-device-map", default=None,
                    help="'auto' shards the SENDER across visible GPUs (same mechanics as "
                         "--receiver-device-map: passed straight to e015.core.load; inputs "
                         "still go to --device, the accelerate hooks route them). Default "
                         "None keeps every existing campaign byte-identical; required only "
                         "for senders larger than one card (P3ROSTER0829 stage-2 "
                         "qwen3.5-397b, ADR-p3stage2-sender-devicemap-20260901). The "
                         "realised placement is printed and written to the summary.")
    ap.add_argument("--donor-scan", type=int, default=64,
                    help="candidates examined per 2-opt improvement step")
    ap.add_argument("--strict-donor-match", action="store_true",
                    help="Require BOTH weight and unwritten-forward norm tolerances; "
                         "maximize role-balanced coverage, with no soft fallback. "
                         "New follow-up only; the default historical matcher is unchanged.")
    ap.add_argument("--assignment", default=None,
                    help="reuse a frozen donor assignment (the §4.4 panel cells "
                         "must share the primary run's donor pairs so the cells "
                         "are comparable); default = this run's own file")
    ap.add_argument("--relax-hard-on-drop", action="store_true",
                    help="DEV/SMOKE ONLY: if the hard-clause derangement drops "
                         "episodes, re-match them on entity-disjointness alone "
                         "and flag the rows (donor_hard_relaxed=true)")
    ap.add_argument("--allow-same-family", action="store_true")
    ap.add_argument("--free-gen", dest="free_gen", action="store_true", default=True)
    ap.add_argument("--no-free-gen", dest="free_gen", action="store_false")
    ap.add_argument("--free-max-new-tokens", type=int, default=8)
    ap.add_argument("--free-answer-prefix", default="",
                    help="FREEGEN0906 (dev diagnostic): string appended after the "
                         "free prompt's answer cue (python escapes decoded, so "
                         "'<think>\\n\\n</think>\\n\\n' works); default '' = "
                         "byte-identical prompts to every earlier run")
    ap.add_argument("--free-arms", default="none,text_oracle,MC,MS,PC,PS,text_packet",
                    help="arms that also get the free-generation endpoint")
    ap.add_argument("--free-templates", default="0",
                    help="P4: comma list of frozen free-gen template indices "
                         "(the preregistration note). Template 0 is "
                         "the dev-era prompt and stays on the main arm row; "
                         "the others emit ADDITIONAL row_type=freegen rows")
    ap.add_argument("--free-rollouts", type=int, default=0,
                    help="P4 sensitivity: N sampled rollouts per template at "
                         "--free-temp (0 = greedy only)")
    ap.add_argument("--free-temp", type=float, default=0.7)
    ap.add_argument("--free-rollout-seeds", default=None,
                    help="comma list of base seeds, one per rollout (required "
                         "if --free-rollouts > 0; lock blanks_registry). "
                         "seed_r = base_seed_r XOR crc32(episode|arm|template)")
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--band-taps", default=None,
                    help="EXPLORATORY comma-separated depth fractions replacing the frozen "
                         "BAND_TAPS (e.g. '0.5' for a single-layer write). Development bank only; "
                         "the taps used are recorded on every row as `band`.")
    ap.add_argument("--record-site-norms", action="store_true",
                    help="record, on every write arm's row, the norm the hook ACTUALLY wrote at "
                         "each (layer, position). The per-row delta_norm is a plan-side quantity "
                         "computed against the un-intervened forward: it is flat across the band "
                         "and does not show that the upper layers of a multi-layer write do far "
                         "less than the first. Default off; additive; changes no other field.")
    ap.add_argument("--time-text-gen", type=int, default=0,
                    help="cost axis: time a greedy sender text generation per episode at this byte "
                         "budget (-1 = each episode's realised packet bytes); 0 = off")
    ap.add_argument("--time-text-gen-n", type=int, default=0, help="cap on timed episodes (0 = all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--allow-test", action="store_true",
                    help="required to touch t1_test/t2_test (ONE-TOUCH budget)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if getattr(a, "free_answer_prefix", ""):
        # shell-typed escapes ('<think>\\n\\n</think>\\n\\n') -> real newlines
        a.free_answer_prefix = a.free_answer_prefix.encode().decode("unicode_escape")

    if a.dict_variant == "random":
        print("[demo] NOTE --dict-variant random: the rand_dict arm is then "
              "identical in construction to MC (both use the random dictionary)",
              flush=True)
    sfam, rfam = family(a.sender), family(a.receiver)
    if sfam == rfam and not a.allow_same_family:
        raise SystemExit(
            f"ABORT: {a.sender} and {a.receiver} are the same family ({sfam}). "
            "The demo's claim is CROSS-FAMILY; pass --allow-same-family to run "
            "it as a same-family instrument, and label it as one.")
    if "test" in os.path.basename(a.bank) and not a.allow_test:
        raise SystemExit(
            "ABORT: test banks are a ONE-TOUCH budget "
            "(data/inventory_v2/TEST_SPLIT_MANIFEST.md §7). Pass --allow-test "
            "only inside the PI-signed launch (guard added 2026-08-03; "
            "run_j3final/run_agg2 already had it, run_demo did not).")
    if "test" in os.path.basename(a.bank) and a.n_clusters < 50:
        raise SystemExit("ABORT: refusing a small run on a test bank "
                         "(single-touch discipline, audit §3.9)")

    os.makedirs(a.out, exist_ok=True)
    variant = a.dict_variant
    s_variant = a.sender_dict_variant or variant
    tag = f"{a.sender}__to__{a.receiver}_{variant}_{a.tag}"
    out_path = os.path.join(a.out, f"{tag}.jsonl")
    done = done_keys(out_path)
    dev = a.device

    arms = [x for x in (a.arms.split(",") if a.arms else ARMS)]
    for arm in arms:
        if arm not in ARMS + OPTIONAL_ARMS:
            raise SystemExit(f"unknown arm {arm}")
    for _nf, _ref in (("rand_dict_normfix", "MC"), ("rand_dict_normfix_swapped", "MS")):
        if _nf in arms and _ref not in arms:
            raise SystemExit(f"ABORT: {_nf} matches the amplitude {_ref} actually wrote, so {_ref} "
                             "must be in --arms of the same run (it may resume from rows on disk)")
    free_arms = set(a.free_arms.split(",")) if a.free_gen else set()

    # P4 free-gen replication axis (lock CONFIRM0818): frozen templates plus
    # the sampled-rollout sensitivity. Defaults ("0", 0 rollouts) reproduce
    # the dev-era behaviour byte-for-byte.
    free_templates = [int(x) for x in a.free_templates.split(",")]
    for ti in free_templates:
        if not 0 <= ti < len(FREE_TEMPLATES):
            raise SystemExit(f"unknown free-gen template {ti} (frozen set: "
                             f"0..{len(FREE_TEMPLATES) - 1})")
    if len(set(free_templates)) != len(free_templates):
        raise SystemExit("duplicate index in --free-templates")
    base_seeds = []
    if a.free_rollouts > 0:
        if not a.free_rollout_seeds:
            raise SystemExit("ABORT: --free-rollout-seeds is required when "
                             "--free-rollouts > 0 (the base seeds are lock "
                             "blanks_registry fields, P4)")
        base_seeds = [int(x) for x in a.free_rollout_seeds.split(",")]
        if len(base_seeds) != a.free_rollouts:
            raise SystemExit(f"ABORT: --free-rollout-seeds lists "
                             f"{len(base_seeds)} seeds but --free-rollouts "
                             f"is {a.free_rollouts}")
    # per template: one greedy generation (rollout None) + N sampled rollouts
    free_gen_ids = [None] + list(range(a.free_rollouts))

    inv = json.load(open(a.inventory))
    addr2word = {c["address"]: c["word"] for c in inv["concepts"]}
    word2type = {c["word"]: c["type"] for c in inv["concepts"]}
    facts = json.load(open(a.facts)) if a.facts else {}
    bank_sha = hashlib.sha256(open(a.bank, "rb").read()).hexdigest()[:16]
    code_sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]

    rows = [json.loads(l) for l in open(a.bank)]
    eps = select_episodes(rows, a.n_clusters, a.reps, a.cluster_offset)
    frac_first = assert_balanced(eps)
    twin_frac = sum(e["twin_index"] == 0 for e in eps) / max(len(eps), 1)
    print(f"[demo] {len(eps)} episodes, gold-is-first-alias {frac_first:.3f}, "
          f"twin0 {twin_frac:.3f}", flush=True)

    # ---------------------------------------------------------- SENDER pass
    s_path = dict_path(a.dict_dir, a.sender, s_variant)
    r_path = dict_path(a.dict_dir, a.receiver, variant)
    for p in (s_path, r_path):
        if not os.path.exists(p):
            raise SystemExit(f"ABORT: dictionary not found: {p}")
    SD, swords, s_sha = load_dict(s_path)
    RD, rwords, r_sha = load_dict(r_path)
    if swords != rwords:
        raise SystemExit("ABORT: sender and receiver dictionaries must share "
                         "the canonical ID space (same `words` order)")
    words = rwords
    widx = {w: i for i, w in enumerate(words)}
    types = [word2type.get(w, "unknown") for w in words]
    digest = inventory_digest(words)[:8]     # 4 wire bytes; round-trip exact

    try:
        smodel, stok = load(a.sender, dev, device_map=a.sender_device_map)
    except Exception as e:                    # requirement 1: fail loudly
        raise SenderRequired(
            f"sender {a.sender} failed to load ({e}); the demo cannot "
            "construct packets without it") from e
    s_devices = model_devices(smodel)
    print(f"[demo] sender {a.sender} device_map={a.sender_device_map} "
          f"n_devices={len(s_devices)} devices={s_devices}", flush=True)
    smid = tap_layer(smodel, 0.5)
    SDdev = SD.to(dev) if USE_GPU_SOLVER else SD

    sender_timing: list = []            # cost axis (2026-09-12, additive)

    def sender_solve(evidence: str) -> torch.Tensor:
        """Evidence text -> coefficients over the canonical ids, with the
        SENDER's own dictionary. Same solve entry point as the writer."""
        with _CostTimer() as _tf:
            h = mean_h(smodel, stok, evidence + "\n", smid, dev)
        Dw = SDdev if USE_GPU_SOLVER else SD.cpu()
        with _CostTimer() as _ts:
            a_vec = _solve_a0(h.float().to(Dw.device).unsqueeze(0), Dw, LAM1, LAM2)
        sender_timing.append({"evidence_bytes": len(evidence.encode("utf-8")),
                              "evidence_tokens": int(stok(evidence + "\n", return_tensors="pt").input_ids.shape[1]),
                              "sender_fwd_ms": round(_tf.ms, 3), "sender_solve_ms": round(_ts.ms, 3)})
        return a_vec[0].float().cpu()

    packets, pkt_dropped = build_sender_packets(
        eps, sender_solve, digest, topk=a.topk, qbits=a.qbits)
    # cost axis (2026-09-12, additive; only with --time-text-gen): greedy sender decode of a
    # text message at EACH episode's realised packet byte budget, on the same passage the
    # packet was extracted from (both entities' evidence), timed like the solve above.
    textgen_timing: list = []
    if a.time_text_gen:
        for ep in eps:
            eid = ep["episode_id"]
            if eid not in packets:
                continue
            if a.time_text_gen_n and len(textgen_timing) >= a.time_text_gen_n:
                break
            passage = ep["sender_evidence"]["entity_1"] + " " + ep["sender_evidence"]["entity_2"]
            budget = rate_bytes(packets[eid]) if a.time_text_gen < 0 else int(a.time_text_gen)
            rec = _time_text_gen(smodel, stok, passage, budget, dev)
            rec["episode"] = eid
            textgen_timing.append(rec)
        print(f"[demo] text_gen@B timing: {len(textgen_timing)} episodes", flush=True)
    _timing_path = os.path.join(a.out, f"{a.tag}_timing.json")
    _tsum = {k: _cost_summarize([r[k] for r in sender_timing]) for k in ("sender_fwd_ms", "sender_solve_ms", "evidence_tokens")}
    _gsum = {k: _cost_summarize([r[k] for r in textgen_timing]) for k in ("textgen_ms", "textgen_new_tokens", "textgen_prompt_tokens", "textgen_budget_bytes", "textgen_msg_bytes")} if textgen_timing else {}
    with open(_timing_path, "w") as fh:
        json.dump({"tag": a.tag, "sender": a.sender, "receiver": a.receiver, "environment": _cost_env(),
                   "sender_calls": sender_timing, "sender_summary": _tsum,
                   "text_gen": textgen_timing, "text_gen_summary": _gsum,
                   "note": "cost axis, descriptive; batch 1; first 3 rows of each list are warm-up and excluded from the summaries"},
                  fh, indent=1)
    print(f"[demo] sender timing -> {_timing_path}", flush=True)
    # the closure holds the sender: both references must go before the receiver.
    # gc.collect() is needed for a sharded sender: accelerate's dispatch hooks
    # form module<->forward reference cycles that refcounting alone never
    # frees, so the shards would stay resident until the next GC pass. A
    # no-op numerically on every path.
    del sender_solve, smodel, stok, SDdev
    gc.collect()
    torch.cuda.empty_cache()
    eps = [e for e in eps if e["episode_id"] in packets]
    if not eps:
        raise SystemExit("ABORT: sender produced no packets")
    ent_of = {e["episode_id"]: episode_entities(e, addr2word) for e in eps}
    gold_in_support = {
        eid: [widx.get(ent_of[eid][i]) in packets[eid].lanes[i].concept_ids
              for i in (0, 1)] for eid in packets if eid in ent_of}
    frac_gold = (sum(any(v) for v in gold_in_support.values())
                 / max(len(gold_in_support), 1))
    print(f"[demo] packets: {len(packets)} built, {len(pkt_dropped)} dropped; "
          f"gold word in support for {frac_gold:.3f} of episodes (audit V5 "
          f"split is reproducible from gold_in_support)", flush=True)

    # -------------------------------------------------------- RECEIVER pass
    rmodel, rtok = load(a.receiver, dev, device_map=a.receiver_device_map)
    r_devices = model_devices(rmodel)
    print(f"[demo] receiver {a.receiver} device_map={a.receiver_device_map} "
          f"n_devices={len(r_devices)} devices={r_devices}", flush=True)
    # --band-taps is an EXPLORATORY override (default: the frozen BAND_TAPS). It exists so the
    # question the 2026-09-13 per-layer profile raises -- the write is front-loaded, is the band
    # needed? -- can be asked on the development bank. No gated run may pass it: every frozen lock
    # names the band it froze, and the runner records the taps used on every row.
    _taps = ([float(x) for x in a.band_taps.split(",")] if getattr(a, "band_taps", None)
             else list(BAND_TAPS))
    band = sorted(tap_layer(rmodel, t) for t in _taps)
    if getattr(a, "band_taps", None):
        print(f"[demo] EXPLORATORY band override: taps {_taps} -> layers {band}", flush=True)
    rmid = tap_layer(rmodel, 0.5)
    RDdev = RD.to(dev) if USE_GPU_SOLVER else RD
    gamma = a.gamma if a.gamma is not None else GAMMA.get(a.receiver)
    if gamma is None and a.amp_override is None:
        raise SystemExit(f"ABORT: no frozen gamma for {a.receiver}; pass "
                         "--gamma or --amp-override")

    r_m = None
    if a.amp_override is None:
        mags = []
        for e in [ent_of[ep["episode_id"]][0] for ep in eps[:a.n_mag]]:
            if e not in widx:
                continue
            h = mean_h(rmodel, rtok, LITERAL.format(e=e), rmid, dev).cpu()
            av = _solver.extract(h, RD, lam1=LAM1, lam2=LAM2).a
            if float(av.max()) > 1e-6:
                mags.append(float(av.max()))
        if not mags:
            raise SystemExit("ABORT: no successful magnitude solves; pass "
                             "--amp-override (rank-1/degenerate dictionary)")
        r_m = float(torch.tensor(mags).median())
        mass = gamma * r_m
    else:
        mass = a.amp_override
    scale = mass_scaler(a.qbits, mass)
    print(f"[demo] receiver {a.receiver} band={band} gamma={gamma} m={r_m} "
          f"L1 mass={mass:.3f} dict={variant}", flush=True)

    # random-dictionary control (always available, independent of --dict-variant)
    rnd_path = dict_path(a.dict_dir, a.receiver, "random")
    RND = RND_sha = None
    if any(x in arms for x in RAND_ARMS):
        if not os.path.exists(rnd_path):
            raise SystemExit(f"ABORT: rand_dict arm needs {rnd_path} "
                             "(python3 -m syncomm.build_alt_dicts ...)")
        RND, rnd_words, RND_sha = load_dict(rnd_path)
        if rnd_words != words:
            raise SystemExit("ABORT: random dictionary has a different ID space")
        RNDdev = RND.to(dev) if USE_GPU_SOLVER else RND

    # ------------------------------------------------- prompts and geometry
    def prompts_for(ep, oracle=False, packet_text=None, free=False,
                    template=0):
        al = ep["public_schema"]["aliases"]
        a1, a2 = al["entity_1"], al["entity_2"]
        m = alias_entity_map(ep, addr2word)
        if oracle:
            s1 = ORACLE.format(a=a1, e=m["entity_1"])
            s2 = ORACLE.format(a=a2, e=m["entity_2"])
        else:
            s1, s2 = CARRIER.format(a=a1), CARRIER.format(a=a2)
        head = f"{s1} {s2}\n"
        if packet_text:
            head += packet_text
        q = ep["receiver_question"]
        if free:
            return free_prompt(head, q, template,
                               getattr(a, "free_answer_prefix", "") or ""), (s1, s2)
        return FEWSHOT_T2 + head + f"Q: {q}\nA:", (s1, s2)

    base_prompt, base_pos, ids_ok = {}, {}, []
    for ep in eps:
        eid = ep["episode_id"]
        p, (s1, s2) = prompts_for(ep)
        pos = [sent_span(rtok, p, s1), sent_span(rtok, p, s2)]
        gid = first_tok(rtok, ep["forced_choice"]["gold"])
        cid = first_tok(rtok, ep["forced_choice"]["counterfactual"])
        if not pos[0] or not pos[1] or gid is None or cid is None or gid == cid:
            continue
        base_prompt[eid], base_pos[eid] = p, pos
        ids_ok.append(ep)
    eps = ids_ok
    print(f"[demo] {len(eps)} episodes with usable spans and alias tokens",
          flush=True)

    # a0 / ||h|| at the write sites, primary receiver dictionary (used by the
    # donor matcher and logged on every row).
    site_cache = {}

    def sites_for(eid, Dw):
        kk = (eid, id(Dw))
        if kk not in site_cache:
            site_cache[kk] = measure_sites(rmodel, rtok, base_prompt[eid], band,
                                           base_pos[eid], Dw, dev)
        return site_cache[kk]

    # (reference arm, episode) -> {(layer, position): the norm that arm really wrote. Persisted on
    # the reference row so that a resumed run (its rows already in `done`, hence skipped) can still
    # match the control exactly. Each control arm matches its OWN real counterpart: the correct-slot
    # control matches MC, the exchanged-slot control matches MS, so both legs of the control
    # interaction are matched to the leg of the real interaction they stand against.
    mc_applied_norms: dict = {}
    want_norms = any(x in NORMFIX_ARMS for x in arms)
    # --record-site-norms turns the same recording on for EVERY write arm, which is how the
    # three-layer band's actual per-layer write is measured (the plan-side norm every row logs is
    # flat across the band and hides it). Default off, so no frozen runner changes behaviour.
    record_all = bool(getattr(a, "record_site_norms", False))
    if want_norms:
        try:
            with open(out_path) as _f:
                for _line in _f:
                    try:
                        _r = json.loads(_line)
                    except Exception:
                        continue
                    if (_r.get("arm") in NORMFIX_REFERENCE.values()
                            and _r.get("site_norms_applied")):
                        mc_applied_norms[(_r["arm"], _r["episode"])] = {
                            (int(k.split("|")[0]), int(k.split("|")[1])): float(v)
                            for k, v in _r["site_norms_applied"].items()}
        except FileNotFoundError:
            pass
        if mc_applied_norms:
            print(f"[normfix] recovered reference site norms for {len(mc_applied_norms)} "
                  "(arm, episode) pairs from rows already on disk", flush=True)
    for ep in eps:
        sites_for(ep["episode_id"], RDdev)

    # ------------------------------------------------- donor assignment (N2)
    asg_path = a.assignment or os.path.join(a.out, f"{tag}_assignment.json")
    eids = [e["episode_id"] for e in eps]
    eids_sha = hashlib.sha256("\n".join(eids).encode()).hexdigest()[:16]
    attr_of = {e["episode_id"]: e["provenance"].get("attribute") for e in eps}

    def self_norm(eid):
        s, A0, hn = sites_for(eid, RDdev)
        return delta_stats(s, A0, hn, RDdev,
                           build_plan(packets[eid], False, scale))[0]

    selfn = {eid: self_norm(eid) for eid in eids}

    def donor_norm(rec, don):
        s, A0, hn = sites_for(rec, RDdev)
        return delta_stats(s, A0, hn, RDdev,
                           build_plan(packets[don], False, scale))[0]

    _cost_cache = {}

    def cost(rec, don):
        if (rec, don) not in _cost_cache:
            qd = q_max_diff(packets[rec], packets[don])
            rd = delta_rel_diff(selfn[rec], donor_norm(rec, don))
            _cost_cache[(rec, don)] = (donor_cost(qd, rd, a.q_tol, a.delta_tol),
                                       qd, rd)
        return _cost_cache[(rec, don)][0]

    assignment, dropped, relaxed = {}, [], set()
    donor_qd, donor_rel = {}, {}
    if any(x in arms for x in DONOR_ARMS):
        cached = None
        if os.path.exists(asg_path):
            cached = json.load(open(asg_path))
            if cached.get("episodes_sha") != eids_sha:
                cached = None
        if cached:
            if bool(cached.get("strict_donor_match", False)) != a.strict_donor_match:
                raise SystemExit("ABORT: cached donor assignment uses a different matching rule")
            assignment = cached["assignment"]
            dropped = cached.get("dropped", [])
            relaxed = set(cached.get("relaxed", []))
            print(f"[demo] reusing frozen assignment {asg_path}", flush=True)
            if cached.get("sender_dict_variant") not in (None, s_variant):
                print(f"[demo] !! the frozen assignment was matched on "
                      f"{cached['sender_dict_variant']} packets, this run "
                      f"sends {s_variant} packets — the donor pairs are held "
                      "fixed for comparability; the realised mismatch below is "
                      "the number to report", flush=True)
        else:
            feasible = {}
            for r in eids:
                cand = []
                for d in eids:
                    if d == r:
                        continue
                    ok, _why = donor_admissible(
                        packets[r], packets[d], types=types,
                        rec_ents=ent_of[r], don_ents=ent_of[d],
                        attribute=attr_of[r], facts=facts,
                        rec_ent_ids=[widx.get(x) for x in ent_of[r]])
                    if ok and a.strict_donor_match:
                        # Screen cheaply on weights before touching receiver geometry.
                        if q_max_diff(packets[r], packets[d]) > a.q_tol:
                            continue
                        cost(r, d)
                        if _cost_cache[(r, d)][2] > a.delta_tol:
                            continue
                    if ok:
                        cand.append(d)
                # order by the cheap soft clause so the greedy matching starts
                # near the argmin; ||delta|| enters through the 2-opt pass
                feasible[r] = sorted(cand,
                                     key=lambda d, r=r: (
                                         q_max_diff(packets[r], packets[d]), d))
            if a.strict_donor_match:
                if a.relax_hard_on_drop:
                    raise SystemExit("ABORT: strict matching cannot relax hard constraints")
                from .strict_matching import strict_derangement
                assignment, dropped = strict_derangement(eids, feasible, cost)
            else:
                assignment, dropped = role_balanced_derangement(
                    eids, feasible, cost, seed=a.seed, scan=a.donor_scan)
            if dropped and a.relax_hard_on_drop:
                loose = {}
                for r in dropped:
                    rid = {widx.get(x) for x in ent_of[r]} - {None}
                    loose[r] = [
                        d for d in eids if d != r
                        and not (set(ent_of[r]) & set(ent_of[d]))
                        and not implies_same_answer(
                            ent_of[r], ent_of[d], attr_of[r], facts)
                        and not (rid & {i for ln in packets[d].lanes
                                        for i in ln.concept_ids})]
                extra, still = role_balanced_derangement(
                    dropped, loose, cost, seed=a.seed)
                assignment.update(extra)
                relaxed = set(extra)
                dropped = still
                print(f"[demo] RELAXED hard clauses for {len(extra)} episodes "
                      "(dev/smoke only; rows flagged donor_hard_relaxed)",
                      flush=True)
            json.dump({"episodes_sha": eids_sha, "seed": a.seed,
                       "assignment": assignment, "dropped": dropped,
                       "relaxed": sorted(relaxed),
                       "sender": a.sender, "receiver": a.receiver,
                       "sender_dict_variant": s_variant,
                       "dict_variant": variant,
                       "strict_donor_match": a.strict_donor_match,
                       "q_tol": a.q_tol, "delta_tol": a.delta_tol},
                      open(asg_path, "w"), indent=1)
        # identification assertions (defect N2)
        assert all(v != k for k, v in assignment.items()), "fixed point in derangement"
        assert len(set(assignment.values())) == len(assignment), "donor reused"
        if assignment:
            assert set(assignment.values()) == set(assignment), \
                "role balance broken: donor set != recipient set"
        # realised distributions of §3.4 criteria 3 and 5, logged per row and
        # reported in the run note ("if no donor qualifies, take the argmin and
        # record the realised mismatch")
        donor_qd = {r: q_max_diff(packets[r], packets[d])
                    for r, d in assignment.items()}
        donor_rel = {r: delta_rel_diff(selfn[r], donor_norm(r, d))
                     for r, d in assignment.items()}
        qds, rds = list(donor_qd.values()), list(donor_rel.values())
        n_q = sum(1 for x in qds if x <= a.q_tol)
        n_d = sum(1 for x in rds if x <= a.delta_tol)
        if a.strict_donor_match:
            assert n_q == n_d == len(assignment), "strict donor tolerances violated"
        print(f"[demo] donor assignment: {len(assignment)} matched, "
              f"{len(dropped)} dropped; q within {a.q_tol}: {n_q}/{len(qds)}; "
              f"||delta|| within {a.delta_tol:.0%}: {n_d}/{len(rds)}", flush=True)

    # ------------------------------------------------------------- scoring
    wire_failures = 0
    n_rows = 0
    for arm in arms:
        for ep in eps:
            eid = ep["episode_id"]
            key = f"{eid}|{arm}|{variant}"
            # P4: ADDITIONAL free-gen (template, rollout) units owed on this
            # (episode, arm). Template-0 greedy stays ON the main arm row
            # (identity), so with default flags this list is always empty and
            # the loop is the dev-era loop.
            free_units = []
            if arm in free_arms:
                for ti in free_templates:
                    for ri in free_gen_ids:
                        if ti == 0 and ri is None:
                            continue          # lives on the main arm row
                        fk = free_key(key, ti, ri)
                        if fk not in done:
                            free_units.append((ti, ri, fk))
            if key in done and not free_units:
                continue
            if arm in DONOR_ARMS and eid not in assignment:
                continue
            al = ep["public_schema"]["aliases"]
            a1, a2 = al["entity_1"], al["entity_2"]
            e1, e2 = ent_of[eid]
            gold = ep["forced_choice"]["gold"]
            cf = ep["forced_choice"]["counterfactual"]
            gid, cid = first_tok(rtok, gold), first_tok(rtok, cf)

            # ---- which packet is ACTUALLY used (metadata is derived from it,
            # ---- never from the matched packet — audit F17)
            donor = assignment.get(eid) if arm in DONOR_ARMS else None
            if arm in ("HC", "HS"):
                if e1 not in widx or e2 not in widx:
                    continue
                pkt = Packet(digest, a.qbits,
                             (Lane(0, (widx[e1],), (255,) if a.qbits else (0,)),
                              Lane(1, (widx[e2],), (255,) if a.qbits else (0,))))
            elif donor is not None:
                pkt = packets[donor]
            else:
                pkt = packets[eid]
            try:
                pkt_rx = round_trip(pkt)          # the ONLY object consumed
            except ValueError as e:
                wire_failures += 1
                print(f"[demo] WIRE FAILURE {eid} {arm}: {e}", flush=True)
                continue

            swapped = arm in SWAPPED_ARMS
            Dw = RDdev
            dict_used = variant
            if arm in RAND_ARMS:
                Dw, dict_used = RNDdev, "random"
            plan = None
            vec = None
            if arm in ("MC", "MS", "PC", "PS", "rand_dict",
                       "rand_dict_swapped", "HC", "HS") or arm in NORMFIX_ARMS:
                plan = build_plan(pkt_rx, swapped, scale)
            elif arm == "zero_packet":
                zero = Packet(digest, a.qbits, tuple(
                    Lane(ln.slot_id, ln.concept_ids, tuple(0 for _ in ln.q))
                    for ln in pkt_rx.lanes))
                zero = round_trip(zero)
                v = reconstruct(RD, zero, norm_policy="none")
                if float(v.abs().max()) != 0.0:
                    raise SystemExit("ABORT: zero packet did not reconstruct to "
                                     "exactly zero — the no-op arm is not a no-op")
                vec = [v[0].to(dev), v[1].to(dev)]

            packet_text = None
            if arm == "text_packet":
                packet_text = render_text(pkt_rx, words,
                                          slot_names={0: a1, 1: a2})

            # ---- P4 additional free-gen rows: same packet/plan/hook
            # machinery as the main row's generation, one slim row per
            # (template, rollout) unit, each with its own resume key.
            for ti, ri, fk in free_units:
                fp, (fs1, fs2) = prompts_for(ep, oracle=(arm == "text_oracle"),
                                             packet_text=packet_text,
                                             free=True, template=ti)
                fpos = [sent_span(rtok, fp, fs1), sent_span(rtok, fp, fs2)]
                seed_r = (None if ri is None else
                          derive_rollout_seed(base_seeds[ri], eid, arm, ti))
                ftxt = free_generate(rmodel, rtok, fp, dev, band=band, D=Dw,
                                     plan=plan, span_positions=fpos, vec=vec,
                                     max_new_tokens=a.free_max_new_tokens,
                                     temperature=(None if ri is None
                                                  else a.free_temp),
                                     gen_seed=seed_r)
                fans, fok = score_free(ftxt, gold, cf)
                append_jsonl(out_path, {
                    "key": fk, "row_type": "freegen",
                    "cluster": ep["cluster_id"], "episode": eid,
                    "arm": arm, "role": ROLE[arm],
                    "dict_variant": dict_used, "run_dict_variant": variant,
                    "template_idx": ti, "rollout_idx": ri,
                    "temperature": None if ri is None else a.free_temp,
                    "rollout_seed": seed_r,
                    "gold": gold, "cf": cf,
                    "free_text": ftxt, "free_answered": fans,
                    "free_correct": fok,
                    "packet_sha256": packet_sha256(pkt_rx),
                    "seed": a.seed, "code_sha256": code_sha})
                done.add(fk)
                n_rows += 1
            if key in done:
                continue

            prompt, (s1, s2) = prompts_for(ep, oracle=(arm == "text_oracle"),
                                           packet_text=packet_text)
            pos = (base_pos[eid] if arm not in ("text_oracle", "text_packet")
                   else [sent_span(rtok, prompt, s1), sent_span(rtok, prompt, s2)])

            # 2026-09-13: the norm-matched control writes the RANDOM dictionary's direction at the
            # magnitude the REAL dictionary's write would have had on this episode, site by site.
            # Targets are measured INSIDE the reference forward, not by delta_stats' proxy.
            tnorms, rec = norm_control_for_arm(
                arm, eid, plan, mc_applied_norms,
                want_norms=want_norms, record_all=record_all)
            gp, extra = forced_gap(rmodel, rtok, prompt, gid, cid, dev,
                                   band=band, D=Dw, plan=plan,
                                   span_positions=pos, vec=vec, target_norms=tnorms,
                                   record_norms=rec)
            if rec:
                mc_applied_norms[(arm, eid)] = dict(rec)
            if arm in NORMFIX_ARMS and not extra.get("n_rescaled"):
                raise SystemExit(f"ABORT: {arm} rescaled no site on {eid} -- the arm would be "
                                 "identical to rand_dict and the control vacuous")

            # realised write geometry, per row (audit F24)
            if plan is not None:
                s_, A0_, hn_ = sites_for(eid, Dw)
                dn, dbl, hnm, doh = delta_stats(s_, A0_, hn_, Dw, plan)
            elif vec is not None:
                s_, A0_, hn_ = sites_for(eid, RDdev)
                dn, dbl, hnm, doh = (0.0, {}, float(hn_.mean()) if hn_ is not None
                                     else 0.0, 0.0)
            else:
                dn, dbl, hnm, doh = 0.0, {}, 0.0, 0.0

            free_text = free_ans = free_ok = None
            if arm in free_arms:
                fp, (fs1, fs2) = prompts_for(ep, oracle=(arm == "text_oracle"),
                                             packet_text=packet_text, free=True)
                fpos = [sent_span(rtok, fp, fs1), sent_span(rtok, fp, fs2)]
                free_text = free_generate(
                    rmodel, rtok, fp, dev, band=band, D=Dw, plan=plan,
                    span_positions=fpos, vec=vec,
                    max_new_tokens=a.free_max_new_tokens)
                free_ans, free_ok = score_free(free_text, gold, cf)

            lanes_meta = []
            for ln in pkt_rx.lanes:
                lanes_meta.append({
                    "slot_id": ln.slot_id, "k": len(ln.concept_ids),
                    "concept_ids": list(ln.concept_ids),
                    "concept_words": [words[i] for i in ln.concept_ids],
                    "q": list(ln.q),
                    "type_hist": dict(type_hist(ln.concept_ids, types))})
            for li, lm in enumerate(lanes_meta):     # F17 anti-regression
                assert lm["concept_words"] == [
                    words[i] for i in pkt_rx.lanes[li].concept_ids]

            row = {
                "key": key, "cluster": ep["cluster_id"], "episode": eid,
                "split": ep["provenance"].get("split"), "bank_sha256": bank_sha,
                "bank": os.path.basename(a.bank),
                "sender": a.sender, "receiver": a.receiver,
                "sender_family": sfam, "receiver_family": rfam,
                "cross_family": sfam != rfam,
                "arm": arm, "role": ROLE[arm],
                "slots_swapped": swapped,
                "dict_variant": dict_used,       # dictionary actually written
                "run_dict_variant": variant,     # the panel cell (§4.4)
                "sender_dict_variant": s_variant,
                "dict_sha256": (RND_sha if arm in RAND_ARMS else r_sha),
                "sender_dict_sha256": s_sha,
                "inventory_digest": digest,
                "paraphrase_idx": ep["paraphrase_idx"],
                "option_order": ep["option_order"],
                "twin_index": ep["twin_index"],
                "aliases": [a1, a2], "entities": [e1, e2],
                "gold": gold, "cf": cf,
                "attribute": attr_of.get(eid),
                "packet_source": ("oracle_canonical" if arm in ("HC", "HS")
                                  else "sender_extract"),
                "packet_hex": packet_hex(pkt_rx),
                "packet_sha256": packet_sha256(pkt_rx),
                "packet_bytes": rate_bytes(pkt_rx), "qbits": pkt_rx.qbits,
                "lanes": lanes_meta,
                # audit V5 split: did the sender's own extraction recover the
                # entity word? (property of the episode's OWN packet, so it is
                # the same on every arm and joins cleanly)
                "own_gold_in_support": gold_in_support.get(eid),
                # contamination check on the packet ACTUALLY written: a donor
                # that names one of the recipient's entities is not "wrong
                # content" (defect N1, support-level)
                "used_names_recipient_entity": any(
                    widx.get(e) in ln.concept_ids
                    for ln in pkt_rx.lanes for e in (e1, e2)),
                "donor_episode": donor,
                "donor_entities": list(ent_of[donor]) if donor else None,
                "donor_q_maxdiff": donor_qd.get(eid) if donor else None,
                "donor_delta_rel": donor_rel.get(eid) if donor else None,
                "donor_hard_relaxed": (eid in relaxed) if donor else None,
                "self_delta_norm": selfn.get(eid),
                "delta_norm": dn, "delta_norm_by_layer": dbl,
                "h_norm": hnm, "delta_over_h": doh,
                # what the hook really wrote at each site, "<layer>|<position>" -> ||delta||.
                # Present ONLY in a run that has a norm-matched control (so the control can be
                # matched site by site, and a resumed run can recover the targets from disk);
                # absent otherwise, which keeps every other run's row schema exactly as it was.
                **({"site_norms_applied": (None if not rec else
                                           {f"{li}|{p}": v for (li, p), v in sorted(rec.items())})}
                   if (want_norms or record_all) else {}),
                "write_positions": pos, "layers": band,
                "receiver_gamma": gamma, "receiver_m": r_m,
                "l1_mass": mass, "amp_override": a.amp_override,
                "gap": gp, "correct": gp > 0,
                "free_text": free_text, "free_answered": free_ans,
                "free_correct": free_ok,
                "text_bytes": (len(packet_text.encode()) if packet_text else None),
                "seed": a.seed, "code_sha256": code_sha,
                **extra}
            append_jsonl(out_path, row)
            done.add(key)
            n_rows += 1
        print(f"[demo] {tag} {arm} done ({n_rows} new rows)", flush=True)

    summary = {
        "tag": tag, "sender": a.sender, "receiver": a.receiver,
        "cross_family": sfam != rfam, "dict_variant": variant,
        "sender_dict_variant": s_variant, "bank": a.bank,
        "bank_sha256": bank_sha, "code_sha256": code_sha,
        "n_episodes": len(eps), "n_new_rows": n_rows, "arms": arms,
        "gold_is_first_alias": frac_first, "twin0_fraction": twin_frac,
        "packets_built": len(packets), "packets_dropped": len(pkt_dropped),
        "frac_gold_word_in_support": frac_gold,
        "donor_matched": len(assignment), "donor_dropped": len(dropped),
        "donor_relaxed": len(relaxed), "assignment_file": asg_path,
        "wire_failures": wire_failures,
        "gamma": gamma, "receiver_m": r_m, "l1_mass": mass,
        "band": band, "seed": a.seed,
        # realised placement (ADR-p3stage2-sender-devicemap-20260901: the
        # equivalence control's sharded arm is VOID unless n_devices > 1)
        "sender_device_map": a.sender_device_map,
        "sender_n_devices": len(s_devices), "sender_devices": s_devices,
        "receiver_device_map": a.receiver_device_map,
        "receiver_n_devices": len(r_devices), "receiver_devices": r_devices}
    json.dump(summary, open(os.path.join(a.out, f"{tag}_summary.json"), "w"),
              indent=1)
    if wire_failures:
        print(f"[demo] !! wire_failures={wire_failures} (must be 0, §3.7 P7)",
              flush=True)
    print(f"[demo] ALL DONE {tag}: {n_rows} new rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
