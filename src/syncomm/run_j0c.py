"""J0-C known-positive: can a slot write causally install a usable hidden-state?

Self-link (sender = receiver), T1 function-broadcast episodes (inventory v2).
Primary endpoint: forced-choice logit gap logit(gold) - logit(cf) at the answer
position, one forward per (episode, arm) — no generation, no coverage issues.

Arms:
  none          neutral carrier ("The hidden object is something.")
  text_oracle   carrier names the entity (full-information positive control)
  global_add    e015-style all-position, all-chunk addition (diagnostic baseline)
  replace       SlotWrite subtract-add coefficient replacement at the carrier token
  random_coord  replace with support swapped to random other entities
  cf_coord      replace with the FOIL entity's coefficients (predicts gap flip)

Coefficient source (declared): literal sentence "Let me tell you about the {e}."
mean residual (skip first 2 positions) solved against the centered inv2
dictionary; support = top-k active. Evidence-text extraction is reported as a
diagnostic column (extraction_recall), not used for the known-positive arms.

Pilot mode (--arms none,text_oracle) needs no dictionary and selects the prompt
format (lock addendum) before the full run.

  python3 -m syncomm.run_j0c --model qwen3-8b-base --bank data/inventory_v2/t1_dev.jsonl \
      --inventory data/inventory_v2/inventory.json --dict artifacts/jlens2/qwen3-8b-base_tap0.5_inv2.pt \
      --arms none,text_oracle --formats plain,fewshot,options --n-clusters 60 --out artifacts/j0c
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

from .e015.core import (load, tap_layer, residual_sigma, single_token_id,
                        append_jsonl, done_keys)
from .e015.core import Steer
from .writer import SlotWrite
from . import solver as _solver

CARRIER_NONE = "There is a hidden object. The hidden object is something.\n"
CARRIER_ORACLE = "There is a hidden object. The hidden object is a {e}.\n"
LITERAL = "Let me tell you about the {e}. The {e} is one of my favorite things.\n"
FEWSHOT = ("There is a hidden object. The hidden object is a violin.\n"
           "Q: What is the hidden object mainly made of? Answer with one word.\nA: wood\n\n"
           "There is a hidden object. The hidden object is a pigeon.\n"
           "Q: How does the hidden object mainly get around? Answer with one word.\nA: fly\n\n")


def build_prompt(ep, fmt, oracle_entity=None):
    carrier = (CARRIER_ORACLE.format(e=oracle_entity) if oracle_entity
               else CARRIER_NONE)
    q = ep["receiver_question"]
    body = f"{carrier}Q: {q}\nA:"
    if fmt == "fewshot":
        body = FEWSHOT + body
    elif fmt == "options":
        o = ep["public_schema"]["answer_options"]
        body = f"{carrier}Q: {q}\nOptions: {o[0]} or {o[1]}.\nA:"
    return body


def carrier_positions(tok, prompt, word="something"):
    i = prompt.rfind(word)
    if i < 0:
        return []
    enc = tok(prompt, return_offsets_mapping=True)
    return [t for t, (a, b) in enumerate(enc["offset_mapping"])
            if a < i + len(word) and b > i]


@torch.no_grad()
def mean_h(model, tok, text, li, dev):
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    h = model(ids, output_hidden_states=True).hidden_states[li + 1][0].float()
    return h[2:].mean(0)


def gap_batch(model, tok, prompts, gold_ids, cf_ids, dev, hook=None):
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids, mask = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    out = model(ids, attention_mask=mask).logits[:, -1].float()
    return [(float(out[i, g] - out[i, c])) for i, (g, c)
            in enumerate(zip(gold_ids, cf_ids))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--dict", default=None)
    ap.add_argument("--arms", default="none,text_oracle")
    ap.add_argument("--formats", default="plain")
    ap.add_argument("--n-clusters", type=int, default=60)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--gammas", default="1.0")
    ap.add_argument("--betas", default="0.5")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"{a.model}_j0c.jsonl")
    done = done_keys(out_path)

    inv = json.load(open(a.inventory))
    addr2word = {c["address"]: c["word"] for c in inv["concepts"]}
    ent_addrs = [c["address"] for c in inv["concepts"] if c["type"] == "entity"]

    eps, seen = [], set()
    for line in open(a.bank):
        r = json.loads(line)
        if r["cluster_id"] not in seen:
            if r["twin_index"] == 0 and r["paraphrase_idx"] == 0 and r["option_order"] == 0:
                seen.add(r["cluster_id"])
                eps.append(r)
        if len(eps) >= a.n_clusters:
            break

    model, tok = load(a.model, "cuda")
    dev = "cuda"
    li = tap_layer(model, 0.5)
    sigma = residual_sigma(model, tok, li, dev)

    D = None
    words = None
    if a.dict:
        blob = torch.load(a.dict, map_location="cpu", weights_only=False)
        D = blob["dirs"].float()
        words = blob["words"]
    widx = {w: i for i, w in enumerate(words)} if words else {}

    _coef_cache = {}

    def coeffs(entity):
        if entity not in _coef_cache:
            h = mean_h(model, tok, LITERAL.format(e=entity), li, dev).cpu()
            res = _solver.extract(h, D)
            av = res.a.clone()
            top = torch.topk(av, a.topk)
            S = [int(i) for i in top.indices if float(av[i]) > 1e-6]
            _coef_cache[entity] = (S, av[S], res)
        return _coef_cache[entity]

    arms = a.arms.split(",")
    fmts = a.formats.split(",")
    gammas = [float(x) for x in a.gammas.split(",")]
    betas = [float(x) for x in a.betas.split(",")]

    for fmt in fmts:
        for arm in arms:
            doses = (gammas if arm in ("replace", "random_coord", "cf_coord")
                     else betas if arm == "global_add" else [0.0])
            for dose in doses:
                todo = [e for e in eps if
                        f"{e['episode_id']}|{arm}|{fmt}|{dose}" not in done]
                for i in range(0, len(todo), a.batch):
                    chunk = todo[i:i + a.batch]
                    prompts, gids, cids, metas = [], [], [], []
                    hooks_pos, hooks_S, hooks_ah = [], [], []
                    for ep in chunk:
                        entity = ep["provenance"]["entity"]
                        foil = ep["provenance"]["foil"]
                        p = build_prompt(ep, fmt,
                                         oracle_entity=entity if arm == "text_oracle" else None)
                        g = single_token_id(tok, ep["forced_choice"]["gold"])
                        c = single_token_id(tok, ep["forced_choice"]["counterfactual"])
                        if g is None or c is None:
                            continue
                        prompts.append(p)
                        gids.append(g)
                        cids.append(c)
                        meta = {"entity": entity, "foil": foil}
                        if arm in ("replace", "random_coord", "cf_coord"):
                            src = foil if arm == "cf_coord" else entity
                            S, ah, res = coeffs(src)
                            meta["support_words"] = [words[j] for j in S]
                            meta["gold_in_support"] = widx.get(src) in S
                            if arm == "random_coord":
                                gseed = int(hashlib.md5(
                                    f"{ep['episode_id']}|rc".encode()).hexdigest()[:8], 16)
                                gen = torch.Generator().manual_seed(gseed)
                                pool = [widx[addr2word[ad]] for ad in ent_addrs
                                        if addr2word[ad] in widx and widx[addr2word[ad]] not in S]
                                perm = torch.randperm(len(pool), generator=gen)[:len(S)]
                                S = [pool[int(j)] for j in perm]
                                meta["support_words"] = [words[j] for j in S]
                            hooks_S.append(S)
                            hooks_ah.append(ah * dose)
                            hooks_pos.append(carrier_positions(tok, p))
                        metas.append(meta)
                    if not prompts:
                        continue
                    if arm in ("replace", "random_coord", "cf_coord"):
                        # per-row support/a_hat -> one SlotWrite per row (B=1)
                        gaps = []
                        for bi, p in enumerate(prompts):
                            with SlotWrite(model, li, [hooks_pos[bi]], "replace",
                                           D=D, support=hooks_S[bi],
                                           a_hat=hooks_ah[bi]):
                                gaps += gap_batch(model, tok, [p],
                                                  [gids[bi]], [cids[bi]], dev)
                    elif arm == "global_add":
                        gaps = []
                        for bi, p in enumerate(prompts):
                            entity = metas[bi]["entity"]
                            S, ah, _ = coeffs(entity)
                            vec = D[S].sum(0)
                            vec = (dose * sigma) * (vec / (vec.norm() + 1e-8))
                            with Steer(model, li, vec.to(dev)):
                                gaps += gap_batch(model, tok, [p],
                                                  [gids[bi]], [cids[bi]], dev)
                    else:
                        gaps = gap_batch(model, tok, prompts, gids, cids, dev)
                    for ep, gp, meta in zip(chunk, gaps, metas):
                        key = f"{ep['episode_id']}|{arm}|{fmt}|{dose}"
                        append_jsonl(out_path, {
                            "key": key, "cluster": ep["cluster_id"],
                            "episode": ep["episode_id"], "arm": arm, "fmt": fmt,
                            "dose": dose, "gap": gp, "correct": gp > 0,
                            "gold": ep["forced_choice"]["gold"],
                            "cf": ep["forced_choice"]["counterfactual"], **meta})
                        done.add(key)
                print(f"[j0c] {a.model} {fmt}/{arm} done", flush=True)
    print("[j0c] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
