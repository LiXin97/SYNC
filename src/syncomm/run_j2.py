"""J2a/J2b: slot-aware binding under the J0-C-validated writer.

Two alias carrier sentences = two slots. Content x slot 2x2 (matched/shuffled
content, correct/swapped slots) gives the preregistered interaction
  I = (MC - MS) - (HC - HS)
on the forced-choice alias logit gap. Plus none / text_oracle / global_add.

Writer geometry is FROZEN from the J0-C repair verdict (sentence span, layer
band, per-model gamma, oracle one-hot coefficients) — gates/j2_binding.lock.json.

  python3 -m syncomm.run_j2 --model qwen3-8b-base --family t2 \
      --bank data/inventory_v2/t2_dev.jsonl --inventory data/inventory_v2/inventory.json \
      --dict artifacts/jlens2/qwen3-8b-base_tap0.5_inv2.pt --gamma 32 --out artifacts/j2
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os

import numpy as np

import torch

from .e015.core import (load, tap_layer, residual_sigma, append_jsonl,
                        done_keys, Steer)
from .writer import SlotWrite
from .run_j0c import mean_h, LITERAL
from . import solver as _solver

CARRIER = "{a} is a hidden item."
ORACLE = "{a} is a {e}."

FEWSHOT_T2 = ("ZEPH is a violin. QUOR is a pigeon.\n"
              "Q: Which entity can fly? Answer with ZEPH or QUOR.\nA: QUOR\n\n"
              "TRAB is a hoe. MYLK is a stool.\n"
              "Q: Which entity is sat on? Answer with TRAB or MYLK.\nA: MYLK\n\n")
# ADDENDUM2 pilot iteration 1: examples must demonstrate the COMPOSED format
# (role -> entity -> property); the alias-answer examples put the oracle at
# chance (0.44/0.49) because base models copy the demonstrated answer type.
FEWSHOT_T3 = ("ZEPH is a violin. QUOR is a pigeon.\n"
              "Fact: QUOR was above ZEPH.\n"
              "Q: Considering the entity that was above the other: how does it "
              "move? Answer with one word.\nA: fly\n\n"
              "TRAB is a hoe. MYLK is a stool.\n"
              "Fact: MYLK supported TRAB.\n"
              "Q: Considering the entity that supported the other: what is it "
              "mainly made of? Answer with one word.\nA: wood\n\n")


T3_OPS = ("material", "used_for", "location", "taste", "habitat", "motion")
OP_Q = {"material": "what is it mainly made of?",
        "used_for": "what is it used for?",
        "location": "where is it usually found?",
        "taste": "what does it taste like?",
        "habitat": "where does it live?",
        "motion": "how does it move?"}


def _ans(v):
    """fact cells are {'answer': w, ...} or a bare string."""
    if isinstance(v, dict):
        return v.get("answer")
    return v


def compose_t3(facts, relf, agent, patient, rel, role="agent"):
    """-> (question, gold_word, cf_word). The public fact fixes the roles; the
    question asks a PROPERTY of the role-selected entity, so answering needs the
    hidden alias->entity mapping too (ADDENDUM2). role='patient' asks about the
    other entity and swaps gold/cf."""
    fa = facts.get(agent, {}).get("operations", {})
    fp = facts.get(patient, {}).get("operations", {})
    past = (relf.get(rel) or {}).get("past_tense") or f"was {rel}"
    for op in T3_OPS:
        va, vp = _ans(fa.get(op)), _ans(fp.get(op))
        if va and vp and va != vp:
            who = (f"the entity that {past} the other" if role == "agent"
                   else f"the entity that the other {past}")
            q = f"Considering {who}: {OP_Q[op]} Answer with one word."
            return (q, va, vp) if role == "agent" else (q, vp, va)
    return None


def first_tok(tok, s):
    ids = tok(" " + s, add_special_tokens=False).input_ids
    return ids[0] if ids else None


def sent_span(tok, prompt, sent):
    i = prompt.rfind(sent)
    if i < 0:
        return []
    enc = tok(prompt, return_offsets_mapping=True)
    return [t for t, (x, y) in enumerate(enc["offset_mapping"])
            if x < i + len(sent) and y > i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=["t2", "t3"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--gamma", type=float, required=True,
                    help="FROZEN gate dose (lock); adjudication uses only this")
    ap.add_argument("--diag-gammas", default="8,16,32",
                    help="diagnostic-only dose panel (ADDENDUM1); never gate-eligible")
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--cluster-offset", type=int, default=0,
                    help="skip the first N clusters in deterministic order (fresh-set confirmation, j2a_confirm.lock)")
    ap.add_argument("--n-mag", type=int, default=40)
    ap.add_argument("--facts", default=None,
                    help="facts_entities.json; required for the T3 composed question")
    ap.add_argument("--arms", default=None, help="restrict arms (pilot uses none,text_oracle)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"{a.model}_{a.family}_j2.jsonl")
    done = done_keys(out_path)

    inv = json.load(open(a.inventory))
    ent_words = [c["word"] for c in inv["concepts"] if c["type"] == "entity"]
    addr2word = {c["address"]: c["word"] for c in inv["concepts"]}

    # ADDENDUM2: balanced sampling over twins and option orders. Taking only
    # index 0 of each made the gold answer the FIRST alias in 199/199 episodes
    # and saturated the none arm at 0.81-1.00.
    by_cluster = collections.defaultdict(list)
    for line in open(a.bank):
        r = json.loads(line)
        if r["paraphrase_idx"] == 0:
            by_cluster[r["cluster_id"]].append(r)
    eps = []
    kept = 0
    for i, (cid, rs) in enumerate(sorted(by_cluster.items())):
        rs.sort(key=lambda r: (r["twin_index"], r["option_order"]))
        # alternate which (twin, order) combination each cluster contributes;
        # the alternation index stays GLOBAL so offset runs keep the balance
        pick = rs[i % len(rs)] if rs else None
        if pick is None:
            continue
        if i < a.cluster_offset:
            continue
        eps.append(pick)
        kept += 1
        if kept >= a.n_clusters:
            break
    if a.family == "t2":
        frac_first = np.mean([e["forced_choice"]["gold"]
                              == e["public_schema"]["aliases"]["entity_1"]
                              for e in eps])
        print(f"[j2] gold-is-first-alias {frac_first:.3f} over {len(eps)} eps",
              flush=True)
        if not 0.45 <= frac_first <= 0.55:
            raise SystemExit(f"ABORT: position confound, gold-is-first "
                             f"{frac_first:.3f} outside [0.45,0.55] (ADDENDUM2)")
    else:
        # composed T3 answers are property words, so the alias-position guard
        # does not apply; the balance that matters is agent-as-first-alias.
        a2w = {c["address"]: c["word"] for c in
               json.load(open(a.inventory))["concepts"]}
        fa = np.mean([a2w.get(e["slots"]["entity_1"]) == e["provenance"]["agent"]
                      for e in eps])
        print(f"[j2] agent-is-first-alias {fa:.3f} over {len(eps)} eps", flush=True)
        if not 0.35 <= fa <= 0.65:
            raise SystemExit(f"ABORT: role-position confound {fa:.3f} (ADDENDUM2)")

    model, tok = load(a.model, "cuda")
    dev = "cuda"
    band = sorted(tap_layer(model, t) for t in (0.4, 0.5, 0.6))
    li_mid = tap_layer(model, 0.5)
    sigma = residual_sigma(model, tok, li_mid, dev)

    blob = torch.load(a.dict, map_location="cpu", weights_only=False)
    D, words = blob["dirs"].float(), blob["words"]
    widx = {w: i for i, w in enumerate(words)}

    mags = []
    for e in [ep["provenance"]["entity_words"][0] for ep in eps[:a.n_mag]]:
        if e not in widx:
            continue
        h = mean_h(model, tok, LITERAL.format(e=e), li_mid, dev).cpu()
        av = _solver.extract(h, D).a
        if float(av.max()) > 1e-6:
            mags.append(float(av.max()))
    m_oracle = float(torch.tensor(mags).median())
    print(f"[j2] {a.model} {a.family} m={m_oracle:.3f}", flush=True)
    diag = [float(x) for x in a.diag_gammas.split(",")]

    facts = json.load(open(a.facts)) if a.facts else {}
    relfacts = (json.load(open(a.facts.replace("facts_entities", "facts_relations")))
                if a.facts else {})
    if a.family == "t3" and not facts:
        raise SystemExit("t3 requires --facts (composed question, ADDENDUM2)")
    if a.family == "t3" and not any(isinstance(v, dict) and "operations" in v
                                    for v in list(facts.values())[:50]):
        raise SystemExit(
            "ABORT: --facts does not look like facts_entities.json (no "
            "'operations' keys in the first 50 values). Passing "
            "facts_relations.json directly defeats the derivation at the "
            "line above and silently zero-rows every episode "
            "(2026-08-03 zero-row incident).")
    fewshot = FEWSHOT_T2 if a.family == "t2" else FEWSHOT_T3

    def gap(prompt, gid, cid, hooks, amp):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        hs = [SlotWrite(model, li, [pos], "replace", D=D, support=[s],
                        a_hat=torch.tensor([amp]))
              for (li, pos, s) in hooks]
        try:
            out = model(ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        return float(out[gid] - out[cid])

    arms = ["none", "text_oracle", "MC", "MS", "HC", "HS", "global_add"]
    if a.arms:
        arms = a.arms.split(",")
    for arm in arms:
      doses = diag if arm in ("MC", "MS", "HC", "HS") else [0.0]
      for gm in doses:
        for ep in eps:
            key = f"{ep['episode_id']}|{arm}|{gm}"
            if key in done:
                continue
            al = ep["public_schema"]["aliases"]
            a1, a2 = al["entity_1"], al["entity_2"]
            # ADDENDUM2 fix 2: the alias->entity mapping MUST come from slots{}
            # (per-twin ground truth). provenance.entity_words is in fixed
            # [positive,negative] order and inverts the mapping on every twin1
            # episode (pilot: oracle 1.000 on twin0 vs 0.033 on twin1).
            e1 = addr2word.get(ep["slots"]["entity_1"])
            e2 = addr2word.get(ep["slots"]["entity_2"])
            if e1 is None or e2 is None:
                continue
            if e1 not in widx or e2 not in widx:
                continue
            gid = first_tok(tok, ep["forced_choice"]["gold"])
            cid = first_tok(tok, ep["forced_choice"]["counterfactual"])
            if gid is None or cid is None or gid == cid:
                continue
            s1, s2 = CARRIER.format(a=a1), CARRIER.format(a=a2)
            if arm == "text_oracle":
                s1, s2 = ORACLE.format(a=a1, e=e1), ORACLE.format(a=a2, e=e2)
            head = f"{s1} {s2}\n"
            question = ep["receiver_question"]
            if a.family == "t3":
                rel = ep["provenance"]["relation_word"]
                ag, pa = ep["provenance"]["agent"], ep["provenance"]["patient"]
                a_ag = a1 if e1 == ag else a2
                a_pa = a2 if a_ag == a1 else a1
                head += f"Fact: {a_ag} {rel} {a_pa}.\n"
                # ADDENDUM2: the bare role question is answerable from the public
                # fact alone. Compose role -> entity -> property so the hidden
                # alias->entity mapping is required.
                role = "agent" if (ep["twin_index"] + ep["option_order"]) % 2 == 0 else "patient"
                comp = compose_t3(facts, relfacts, ag, pa, rel, role)
                if comp is None:
                    continue
                question, gold_w, cf_w = comp
                gid = first_tok(tok, gold_w)
                cid = first_tok(tok, cf_w)
                if gid is None or cid is None or gid == cid:
                    continue
                meta_extra = {"composed_gold": gold_w, "composed_cf": cf_w,
                              "composed_role": role}
            else:
                meta_extra = {}
            prompt = fewshot + head + f"Q: {question}\nA:"

            hooks = []
            meta = {}
            if arm in ("MC", "MS", "HC", "HS"):
                if arm in ("HC", "HS"):
                    g = torch.Generator().manual_seed(
                        int(hashlib.md5(f"{ep['episode_id']}|h".encode()).hexdigest()[:8], 16))
                    pool = [w for w in ent_words if w in widx and w not in (e1, e2)]
                    pick = torch.randperm(len(pool), generator=g)[:2].tolist()
                    c1, c2 = pool[pick[0]], pool[pick[1]]
                else:
                    c1, c2 = e1, e2
                if arm in ("MS", "HS"):
                    c1, c2 = c2, c1
                meta["content"] = [c1, c2]
                p1, p2 = sent_span(tok, prompt, s1), sent_span(tok, prompt, s2)
                for li in band:
                    hooks.append((li, p1, widx[c1]))
                    hooks.append((li, p2, widx[c2]))
                gp = gap(prompt, gid, cid, hooks, m_oracle * gm)
            elif arm == "global_add":
                vec = D[widx[e1]] + D[widx[e2]]
                vec = (0.5 * sigma) * (vec / (vec.norm() + 1e-8))
                with Steer(model, li_mid, vec.to(dev)):
                    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                    out = model(ids).logits[0, -1].float()
                gp = float(out[gid] - out[cid])
            else:
                ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                out = model(ids).logits[0, -1].float()
                gp = float(out[gid] - out[cid])
            append_jsonl(out_path, {
                **meta_extra,
                "key": key, "cluster": ep["cluster_id"],
                "episode": ep["episode_id"], "family": a.family, "arm": arm,
                "dose": gm, "is_gate_dose": (gm == a.gamma),
                "gap": gp, "correct": gp > 0, "entities": [e1, e2],
                "aliases": [a1, a2], "gold": ep["forced_choice"]["gold"],
                **meta})
            done.add(key)
        print(f"[j2] {a.model} {a.family} {arm} g={gm} done", flush=True)
    n_rows = sum(1 for _ in open(out_path)) if os.path.exists(out_path) else 0
    if n_rows == 0:
        raise SystemExit(
            "ABORT instead of ALL DONE: zero rows written — every episode "
            "was silently skipped (2026-08-03 lesson: DONE markers without "
            "row counts are not completion). Check facts/bank/dict shapes.")
    print(f"[j2] ALL DONE rows={n_rows}", flush=True)


if __name__ == "__main__":
    main()
