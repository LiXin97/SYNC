"""J0-C repair-cycle runner (gates/j0_instrument.lock.ADDENDUM2.md).

Oracle one-hot coefficients isolate the WRITER from the extractor; carrier-
sentence span writes + layer band + raised dose grid attack the dilution
diagnosis. Content controls (random/foil oracle) and a shuffled-support
global_add typing arm included.

  python3 -m syncomm.run_j0cr --model qwen3-8b-base --bank data/inventory_v2/t1_dev.jsonl \
      --inventory data/inventory_v2/inventory.json --dict artifacts/jlens2/qwen3-8b-base_tap0.5_inv2.pt \
      --n-clusters 200 --out artifacts/j0c_repair
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

from .e015.core import (load, tap_layer, residual_sigma, single_token_id,
                        append_jsonl, done_keys, Steer)
from .writer import SlotWrite
from . import solver as _solver
from .run_j0c import (CARRIER_NONE, LITERAL, build_prompt, mean_h)

CARRIER_SENT = "The hidden object is something."


def span_positions(tok, prompt, span):
    target = "something" if span == "token" else CARRIER_SENT
    i = prompt.rfind(target)
    if i < 0:
        return []
    enc = tok(prompt, return_offsets_mapping=True)
    return [t for t, (a, b) in enumerate(enc["offset_mapping"])
            if a < i + len(target) and b > i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--gammas", default="4,8,16,32")
    ap.add_argument("--taps", default="0.4,0.5,0.6,band")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-mag", type=int, default=40)
    ap.add_argument("--arms", default=None,
                    help="comma list to restrict arms (J4 uses none,replace_oracle,random_oracle,cf_oracle)")
    ap.add_argument("--tag", default="j0cr", help="output filename tag")
    ap.add_argument("--amp-override", type=float, default=None,
                    help="absolute write amplitude a_hat=amp (J4 ADDENDUM1: matched across dictionaries; skips the per-dict m solve)")
    ap.add_argument("--allow-test", action="store_true",
                    help="required to touch t1_test/t2_test (ONE-TOUCH budget; "
                         "guard added 2026-08-03, adversarial review F4 — this "
                         "runner executes the post-launch SAE campaign and had "
                         "no guard)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if "test" in os.path.basename(a.bank) and not a.allow_test:
        raise SystemExit(
            "ABORT: test banks are a ONE-TOUCH budget "
            "(data/inventory_v2/TEST_SPLIT_MANIFEST.md §7). Pass --allow-test "
            "only inside a PI-signed launch.")

    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, f"{a.model}_{a.tag}.jsonl")
    done = done_keys(out_path)

    inv = json.load(open(a.inventory))
    ent_words = [c["word"] for c in inv["concepts"] if c["type"] == "entity"]

    eps, seen = [], set()
    for line in open(a.bank):
        r = json.loads(line)
        if r["cluster_id"] not in seen and r["twin_index"] == 0 \
                and r["paraphrase_idx"] == 0 and r["option_order"] == 0:
            seen.add(r["cluster_id"])
            eps.append(r)
        if len(eps) >= a.n_clusters:
            break

    model, tok = load(a.model, "cuda")
    dev = "cuda"
    # BUGFIX 2026-07-31: band must not depend on which single taps were
    # requested. With --taps band the old code left taps={} and band=[], so
    # every write arm silently became a no-op byte-identical to `none`
    # (the J4 first launch produced exactly-0.000 contrasts and was discarded).
    BAND_TAPS = (0.4, 0.5, 0.6)
    band = sorted(tap_layer(model, t) for t in BAND_TAPS)
    taps = {}
    for t in a.taps.split(","):
        if t != "band":
            taps[t] = tap_layer(model, float(t))
    li_mid = tap_layer(model, 0.5)
    sigma = residual_sigma(model, tok, li_mid, dev)

    blob = torch.load(a.dict, map_location="cpu", weights_only=False)
    D, words = blob["dirs"].float(), blob["words"]
    widx = {w: i for i, w in enumerate(words)}

    # oracle magnitude m: per-model median of the largest active coefficient
    # over literal solves on the first n dev entities (ADDENDUM2 §1). With
    # --amp-override the per-dict solve is skipped and a_hat = amp/dose * dose
    # = amp exactly (J4 ADDENDUM1 amplitude matching; also removes the rank-1
    # common-dictionary nan path).
    if a.amp_override is not None:
        m_oracle = None
        print(f"[j0cr] {a.model} amp-override={a.amp_override}", flush=True)
    else:
        mags = []
        for e in [ep["provenance"]["entity"] for ep in eps[:a.n_mag]]:
            if e not in widx:
                continue
            h = mean_h(model, tok, LITERAL.format(e=e), li_mid, dev).cpu()
            av = _solver.extract(h, D).a
            if float(av.max()) > 1e-6:
                mags.append(float(av.max()))
        if not mags:
            raise SystemExit("ABORT: no successful magnitude solves and no "
                             "--amp-override (nan write would crash downstream)")
        m_oracle = float(torch.tensor(mags).median())
        print(f"[j0cr] {a.model} oracle magnitude m={m_oracle:.3f} (n={len(mags)})",
              flush=True)

    def gap_one(prompt, gid, cid, hooks):
        enc = tok(prompt, return_tensors="pt")
        ids = enc.input_ids.to(dev)
        hs = [SlotWrite(model, li, [pos], "replace", D=D, support=S, a_hat=ah)
              for (li, pos, S, ah) in hooks]
        try:
            out = model(ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        return float(out[gid] - out[cid])

    arms = ["none", "text_oracle", "replace_oracle", "random_oracle",
            "cf_oracle", "replace_extract", "global_add_shuf", "replace_token"]
    if a.arms:
        arms = [x for x in a.arms.split(",")]
    gammas = [float(x) for x in a.gammas.split(",")]
    tapnames = a.taps.split(",")

    for arm in arms:
        grid = ([("-", "-", 0.0)] if arm in ("none", "text_oracle") else
                [("sent", "0.5", 0.5)] if arm == "global_add_shuf" else
                [("token", "0.5", g) for g in gammas] if arm == "replace_token" else
                [("sent", tn, g) for tn in tapnames for g in gammas])
        for span, tapname, dose in grid:
            for ep in eps:
                key = f"{ep['episode_id']}|{arm}|{span}|{tapname}|{dose}"
                if key in done:
                    continue
                entity = ep["provenance"]["entity"]
                foil = ep["provenance"]["foil"]
                p = build_prompt(ep, "fewshot",
                                 oracle_entity=entity if arm == "text_oracle" else None)
                gid = single_token_id(tok, ep["forced_choice"]["gold"])
                cid = single_token_id(tok, ep["forced_choice"]["counterfactual"])
                if gid is None or cid is None:
                    continue
                meta = {"entity": entity}
                if arm in ("none", "text_oracle"):
                    ids = tok(p, return_tensors="pt").input_ids.to(dev)
                    out = model(ids).logits[0, -1].float()
                    gp = float(out[gid] - out[cid])
                elif arm == "global_add_shuf":
                    gseed = int(hashlib.md5(f"{ep['episode_id']}|gs".encode())
                                .hexdigest()[:8], 16)
                    gen = torch.Generator().manual_seed(gseed)
                    pool = [w for w in ent_words if w in widx and w != entity]
                    w_r = pool[int(torch.randint(len(pool), (1,), generator=gen))]
                    vec = D[widx[w_r]]
                    vec = (dose * sigma) * (vec / (vec.norm() + 1e-8))
                    with Steer(model, li_mid, vec.to(dev)):
                        ids = tok(p, return_tensors="pt").input_ids.to(dev)
                        out = model(ids).logits[0, -1].float()
                    gp = float(out[gid] - out[cid])
                    meta["shuf_word"] = w_r
                else:
                    if arm == "replace_extract":
                        h = mean_h(model, tok, LITERAL.format(e=entity),
                                   li_mid, dev).cpu()
                        av = _solver.extract(h, D).a
                        top = torch.topk(av, 4)
                        S = [int(i) for i in top.indices if float(av[i]) > 1e-6]
                        ah = av[S] * dose / 4.0   # comparable scale to gamma*m
                        meta["gold_in_support"] = widx.get(entity) in S
                    else:
                        src = (foil if arm == "cf_oracle" else entity)
                        if arm == "random_oracle":
                            gseed = int(hashlib.md5(f"{ep['episode_id']}|ro".encode())
                                        .hexdigest()[:8], 16)
                            gen = torch.Generator().manual_seed(gseed)
                            pool = [w for w in ent_words if w in widx
                                    and w not in (entity, foil)]
                            src = pool[int(torch.randint(len(pool), (1,),
                                                         generator=gen))]
                        if src not in widx:
                            continue
                        S = [widx[src]]
                        amp = (a.amp_override if a.amp_override is not None
                               else m_oracle * dose)
                        ah = torch.tensor([amp])
                        meta["src"] = src
                    pos = span_positions(tok, p, span)
                    lis = band if tapname == "band" else [taps[tapname]]
                    hooks = [(li, pos, S, ah) for li in lis]
                    if not hooks or not pos:
                        raise SystemExit(
                            f"ABORT: write arm {arm} produced no hooks "
                            f"(layers={lis}, positions={pos}) — a silent no-op "
                            "would be byte-identical to the control")
                    gp = gap_one(p, gid, cid, hooks)
                append_jsonl(out_path, {
                    "key": key, "cluster": ep["cluster_id"],
                    "episode": ep["episode_id"], "arm": arm, "span": span,
                    "tap": tapname, "dose": dose, "gap": gp, "correct": gp > 0,
                    **meta})
                done.add(key)
            print(f"[j0cr] {a.model} {arm} {span}/{tapname}/{dose} done",
                  flush=True)
    print("[j0cr] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
