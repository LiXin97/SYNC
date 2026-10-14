"""J3-transport (+ J5 co-primary): cross-model coordinate transport on T1.

Sender solves with ITS OWN dictionary; only a serialized packet crosses the
boundary; receiver reconstructs with ITS OWN dictionary and writes at ITS OWN
frozen J0-C configuration. No pair-specific anything (gates/j3_transport.lock.json).

  python3 -m syncomm.run_j3 --sender qwen3-8b-base --receiver llama-3.1-8b \
      --mode oracle --bank data/inventory_v2/t1_dev.jsonl \
      --inventory data/inventory_v2/inventory.json --out artifacts/j3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch

from .e015.core import load, tap_layer, single_token_id, append_jsonl, done_keys
from .writer import SlotWrite
from .run_j0c import build_prompt, mean_h, LITERAL
from .run_j0cr import span_positions
from . import solver as _solver
from .packet import Lane, Packet, inventory_digest, rate_bytes, dequantize

GAMMA = {"qwen3-8b-base": 32.0, "llama-3.1-8b": 16.0,
         "qwen3-14b-base": 32.0, "qwen3-32b": 32.0,
         "glm-4-9b": 32.0, "phi-4": 32.0}
BAND_TAPS = (0.4, 0.5, 0.6)
TOPK = 4
QBITS = 8


def oracle_mag(model, tok, D, li, words_set, entities, dev, n=40):
    mags = []
    for e in entities[:n]:
        if e not in words_set:
            continue
        h = mean_h(model, tok, LITERAL.format(e=e), li, dev).cpu()
        av = _solver.extract(h, D).a
        if float(av.max()) > 1e-6:
            mags.append(float(av.max()))
    return float(torch.tensor(mags).median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sender", required=True)
    ap.add_argument("--receiver", required=True)
    ap.add_argument("--mode", default="oracle", choices=["oracle", "extract"])
    ap.add_argument("--bank", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--dict-dir", default="artifacts/jlens2")
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.sender}__to__{a.receiver}_{a.mode}"
    out_path = os.path.join(a.out, f"{tag}.jsonl")
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

    dev = "cuda"
    rmodel, rtok = load(a.receiver, dev)
    rband = sorted(tap_layer(rmodel, t) for t in BAND_TAPS)
    rmid = tap_layer(rmodel, 0.5)
    rblob = torch.load(os.path.join(a.dict_dir, f"{a.receiver}_tap0.5_inv2.pt"),
                       map_location="cpu", weights_only=False)
    RD, rwords = rblob["dirs"].float(), rblob["words"]
    ridx = {w: i for i, w in enumerate(rwords)}
    ents = [ep["provenance"]["entity"] for ep in eps]
    r_m = oracle_mag(rmodel, rtok, RD, rmid, set(rwords), ents, dev)
    digest = inventory_digest(rwords)
    print(f"[j3] receiver {a.receiver} m={r_m:.3f} gamma={GAMMA[a.receiver]}",
          flush=True)

    # ---- sender side: build packets (canonical IDs are shared addresses) ----
    packets = {}
    if a.mode == "extract":
        smodel, stok = load(a.sender, dev)
        smid = tap_layer(smodel, 0.5)
        sblob = torch.load(os.path.join(a.dict_dir, f"{a.sender}_tap0.5_inv2.pt"),
                           map_location="cpu", weights_only=False)
        SD, swords = sblob["dirs"].float(), sblob["words"]
        assert swords == rwords, "inventories must share the canonical ID space"
        for ep in eps:
            e = ep["provenance"]["entity"]
            if e in packets:
                continue
            ev = ep["sender_evidence"]["entity_1"]
            h = mean_h(smodel, stok, ev + "\n", smid, dev).cpu()
            av = _solver.extract(h, SD).a
            top = torch.topk(av, TOPK)
            S = sorted(int(i) for i in top.indices if float(av[i]) > 1e-6)
            if not S:
                continue
            vals = av[S]
            q = tuple(int(round(255 * float(v / vals.max()))) for v in vals)
            packets[e] = Packet(digest, QBITS, (Lane(0, tuple(S), q),))
        del smodel
        torch.cuda.empty_cache()
    else:
        for ep in eps:
            e = ep["provenance"]["entity"]
            if e in packets or e not in ridx:
                continue
            packets[e] = Packet(digest, QBITS, (Lane(0, (ridx[e],), (255,)),))

    def receiver_write(prompt, gid, cid, S, coeffs):
        pos = span_positions(rtok, prompt, "sent")
        if not pos or not S:
            raise SystemExit("ABORT: empty write (no-op would equal control)")
        hs = [SlotWrite(rmodel, li, [pos], "replace", D=RD, support=list(S),
                        a_hat=coeffs) for li in rband]
        try:
            ids = rtok(prompt, return_tensors="pt").input_ids.to(dev)
            out = rmodel(ids).logits[0, -1].float()
        finally:
            for h in hs:
                h.remove()
        return float(out[gid] - out[cid])

    def scaled(pkt):
        """receiver-side calibration: dequantized coefficients rescaled to the
        receiver's own L1 mass gamma_r * m_r (pairwise-free by construction)."""
        ln = pkt.lanes[0]
        v = dequantize(ln.q, pkt.qbits)
        s = float(v.sum())
        mass = GAMMA[a.receiver] * r_m
        return list(ln.concept_ids), (v * (mass / s) if s > 1e-8 else v)

    arms = ["none", "text_oracle", "inject_matched", "inject_shuffled",
            "inject_cf", "text_packet"]
    for arm in arms:
        for ep in eps:
            key = f"{ep['episode_id']}|{arm}"
            if key in done:
                continue
            e, foil = ep["provenance"]["entity"], ep["provenance"]["foil"]
            if e not in packets:
                continue
            gid = single_token_id(rtok, ep["forced_choice"]["gold"])
            cid = single_token_id(rtok, ep["forced_choice"]["counterfactual"])
            if gid is None or cid is None:
                continue
            pkt = packets[e]
            meta = {"packet_bytes": rate_bytes(pkt),
                    "packet_words": [rwords[i] for i in pkt.lanes[0].concept_ids]}
            if arm in ("inject_shuffled", "inject_cf"):
                if arm == "inject_cf":
                    src = foil
                else:
                    g = torch.Generator().manual_seed(
                        int(hashlib.md5(f"{ep['episode_id']}|sh".encode())
                            .hexdigest()[:8], 16))
                    pool = [w for w in ent_words if w in ridx and w not in (e, foil)]
                    src = pool[int(torch.randint(len(pool), (1,), generator=g))]
                if src not in ridx:
                    continue
                pkt = Packet(digest, QBITS, (Lane(0, (ridx[src],), (255,)),))
                meta["control_src"] = src
            if arm == "text_packet":
                ws = [rwords[i] for i in pkt.lanes[0].concept_ids]
                vs = dequantize(pkt.lanes[0].q, pkt.qbits)
                txt = ", ".join(f"{w} ({float(v):.2f})" for w, v in zip(ws, vs))
                prompt = build_prompt(ep, "fewshot").replace(
                    "The hidden object is something.",
                    f"The hidden object is: {txt}.")
                ids = rtok(prompt, return_tensors="pt").input_ids.to(dev)
                out = rmodel(ids).logits[0, -1].float()
                gp = float(out[gid] - out[cid])
                meta["text_bytes"] = len(txt.encode())
            elif arm in ("none", "text_oracle"):
                prompt = build_prompt(ep, "fewshot",
                                      oracle_entity=e if arm == "text_oracle" else None)
                ids = rtok(prompt, return_tensors="pt").input_ids.to(dev)
                out = rmodel(ids).logits[0, -1].float()
                gp = float(out[gid] - out[cid])
            else:
                prompt = build_prompt(ep, "fewshot")
                S, coeffs = scaled(pkt)
                gp = receiver_write(prompt, gid, cid, S, coeffs)
            append_jsonl(out_path, {
                "key": key, "cluster": ep["cluster_id"],
                "episode": ep["episode_id"], "sender": a.sender,
                "receiver": a.receiver, "mode": a.mode, "arm": arm,
                "gap": gp, "correct": gp > 0, "entity": e, **meta})
            done.add(key)
        print(f"[j3] {tag} {arm} done", flush=True)
    print("[j3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
