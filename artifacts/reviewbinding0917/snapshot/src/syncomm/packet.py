"""Canonical SYNC packet: the ONE packet object every runner consumes.

Refocus C1 (the study protocol
§3): a single reconstruction function v(P) = D[:,S] a_S with an explicit
normalization policy, shared by the injection and text renderers, so "same
packet" is true by construction rather than by prose. Historical context: the
E015-E033 runners serialized (id, q8) pairs but every injection site discarded
q8 (verified 2026-07-30, the run note) —
this module makes that class of divergence structurally impossible.

Wire format (all bytes count toward the rate, audit 02 §7):
  [0]    version (1)
  [1:5]  inventory digest prefix (4 bytes)
  [5]    qbits: coefficient precision in bits; 0 = support-only
  [6]    number of lanes
  per lane:
    [0]  slot_id
    [1]  k (support size)
    [2:2+k]        concept ids, ascending (inventory <= 255 concepts)
    [...]          k coefficients, qbits each, big-endian bit-packed
  [-1]   checksum: xor of all preceding bytes

Tensor orientation: dictionaries are [num_concepts, hidden] everywhere
(the project rules §8); silent transposes forbidden.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

VERSION = 1
EPS = 1e-8


@dataclass(frozen=True)
class Lane:
    slot_id: int
    concept_ids: tuple[int, ...]   # ascending, no duplicates
    q: tuple[int, ...]             # quantized coefficients, len == len(concept_ids)

    def __post_init__(self):
        if len(self.concept_ids) != len(self.q):
            raise ValueError("concept_ids and q length mismatch")
        if list(self.concept_ids) != sorted(set(self.concept_ids)):
            raise ValueError("concept_ids must be ascending and unique")


@dataclass(frozen=True)
class Packet:
    inventory_digest: str          # >= 8 hex chars; first 8 go on the wire
    qbits: int                     # 0 (support-only), or 2/4/8
    lanes: tuple[Lane, ...]
    version: int = VERSION

    def __post_init__(self):
        if self.qbits not in (0, 2, 4, 8):
            raise ValueError(f"unsupported qbits {self.qbits}")
        qmax = (1 << self.qbits) - 1 if self.qbits else 0
        for ln in self.lanes:
            for v in ln.q:
                if not (0 <= v <= max(qmax, 0)):
                    raise ValueError(f"q value {v} out of range for qbits={self.qbits}")


def inventory_digest(words: list[str]) -> str:
    return hashlib.sha256("\n".join(words).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- wire format

def _pack_bits(values: tuple[int, ...], bits: int) -> bytes:
    if bits == 0:
        return b""
    acc = 0
    nacc = 0
    out = bytearray()
    for v in values:
        acc = (acc << bits) | v
        nacc += bits
        while nacc >= 8:
            nacc -= 8
            out.append((acc >> nacc) & 0xFF)
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out)


def _unpack_bits(data: bytes, bits: int, count: int) -> tuple[int, ...]:
    if bits == 0:
        return tuple(0 for _ in range(count))
    acc = 0
    nacc = 0
    vals = []
    it = iter(data)
    for _ in range(count):
        while nacc < bits:
            acc = (acc << 8) | next(it)
            nacc += 8
        nacc -= bits
        vals.append((acc >> nacc) & ((1 << bits) - 1))
    return tuple(vals)


def serialize(p: Packet) -> bytes:
    out = bytearray()
    out.append(p.version)
    out += bytes.fromhex(p.inventory_digest[:8])
    out.append(p.qbits)
    out.append(len(p.lanes))
    for ln in p.lanes:
        out.append(ln.slot_id)
        out.append(len(ln.concept_ids))
        for cid in ln.concept_ids:
            if not 0 <= cid <= 255:
                raise ValueError("concept id out of byte range")
            out.append(cid)
        out += _pack_bits(ln.q, p.qbits)
    chk = 0
    for b in out:
        chk ^= b
    out.append(chk)
    return bytes(out)


def deserialize(data: bytes) -> Packet:
    chk = 0
    for b in data[:-1]:
        chk ^= b
    if chk != data[-1]:
        raise ValueError("checksum mismatch")
    ver = data[0]
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    digest = data[1:5].hex()
    qbits = data[5]
    n_lanes = data[6]
    pos = 7
    lanes = []
    for _ in range(n_lanes):
        slot_id = data[pos]
        k = data[pos + 1]
        pos += 2
        cids = tuple(data[pos:pos + k])
        pos += k
        nbytes = (k * qbits + 7) // 8
        q = _unpack_bits(data[pos:pos + nbytes], qbits, k)
        pos += nbytes
        lanes.append(Lane(slot_id, cids, q))
    return Packet(inventory_digest=digest, qbits=qbits, lanes=tuple(lanes), version=ver)


def rate_bytes(p: Packet) -> int:
    """Actual serialized size. Every transmitted field counts (audit 02 §7)."""
    return len(serialize(p))


# ------------------------------------------------------------- reconstruction

def dequantize(q: tuple[int, ...], qbits: int) -> torch.Tensor:
    """-> coefficients in [0, 1]. Support-only (qbits=0) means uniform 1s:
    a support-only packet and a uniform-q packet reconstruct identically by
    construction (C1 unit-test requirement)."""
    if qbits == 0:
        return torch.ones(len(q), dtype=torch.float32)
    qmax = float((1 << qbits) - 1)
    return torch.tensor(q, dtype=torch.float32) / qmax


def reconstruct(D: torch.Tensor, p: Packet, *, weight_mode: str = "quantized",
                norm_policy: str = "unit", calib=None) -> torch.Tensor:
    """v(P) = D[:,S] a_S per lane. D is [num_concepts, hidden]; returns
    [n_lanes, hidden], fp32, on D's device.

    weight_mode: 'quantized' uses dequantized q (then per-model `calib`
      callable, identity if None); 'uniform' forces a_i = 1 (the support-only
      arm, regardless of transmitted q).
    norm_policy: 'unit' L2-normalizes each lane vector (the historical E033
      convention; dose applied downstream as beta*sigma); 'none' returns the
      raw weighted sum (coefficient-magnitude-preserving writers).
    A packet with all-zero coefficients under 'quantized'/'none' reconstructs
    to exactly zero (zero-packet no-op requirement).
    """
    if D.dim() != 2:
        raise ValueError("D must be [num_concepts, hidden]")
    out = torch.zeros(len(p.lanes), D.shape[1], dtype=torch.float32, device=D.device)
    for li, ln in enumerate(p.lanes):
        if weight_mode == "uniform":
            a = torch.ones(len(ln.concept_ids), dtype=torch.float32)
        elif weight_mode == "quantized":
            a = dequantize(ln.q, p.qbits)
            if calib is not None:
                a = calib(a)
        else:
            raise ValueError(f"unknown weight_mode {weight_mode}")
        a = a.to(D.device)
        idx = torch.tensor(ln.concept_ids, dtype=torch.long, device=D.device)
        v = (a.unsqueeze(1) * D[idx].float()).sum(0)
        if norm_policy == "unit":
            n = v.norm()
            if n > EPS:
                v = v / n
        elif norm_policy != "none":
            raise ValueError(f"unknown norm_policy {norm_policy}")
        out[li] = v
    return out


# ------------------------------------------------------------------ renderers

def render_text(p: Packet, vocab: list[str], *, weights: bool = True,
                slot_names: dict[int, str] | None = None) -> str:
    """Text rendering of the SAME packet object the injection path consumes.
    weights=False renders support-only (must be paired with weight_mode=
    'uniform' on the injection side for a matched support-only 2x2)."""
    parts = []
    for ln in p.lanes:
        a = dequantize(ln.q, p.qbits)
        items = ", ".join(
            f"{vocab[cid]} ({float(av):.2f})" if weights and p.qbits else vocab[cid]
            for cid, av in zip(ln.concept_ids, a))
        if slot_names and ln.slot_id in slot_names:
            parts.append(f"{slot_names[ln.slot_id]}: {items}")
        else:
            parts.append(items)
    return "The secret topic is: " + "; ".join(parts) + ".\n"


# ------------------------------------------------------------------ controls

def shuffle_q(p: Packet, gen: torch.Generator) -> Packet:
    """Permute coefficients within each lane; support and rate unchanged."""
    lanes = []
    for ln in p.lanes:
        perm = torch.randperm(len(ln.q), generator=gen).tolist()
        lanes.append(Lane(ln.slot_id, ln.concept_ids,
                          tuple(ln.q[i] for i in perm)))
    return Packet(p.inventory_digest, p.qbits, tuple(lanes), p.version)


def shuffle_support(p: Packet, candidate_ids: list[int], gen: torch.Generator) -> Packet:
    """Replace each lane's support with a same-size draw from candidate_ids
    (excluding the original support); the coefficient multiset is preserved."""
    lanes = []
    for ln in p.lanes:
        pool = [c for c in candidate_ids if c not in ln.concept_ids]
        if len(pool) < len(ln.concept_ids):
            raise ValueError("candidate pool too small")
        pick = torch.randperm(len(pool), generator=gen)[:len(ln.concept_ids)].tolist()
        new_ids = sorted(pool[i] for i in pick)
        lanes.append(Lane(ln.slot_id, tuple(new_ids), ln.q))
    return Packet(p.inventory_digest, p.qbits, tuple(lanes), p.version)


def merge_packets(packets: list[Packet], *, mode: str = "max") -> Packet:
    """Fan-in merge (ADR-0008 BUILD 1): N senders' packets -> one packet.

    Lanes with the same slot_id are merged by UNION of concept ids; where the
    same id appears in several packets its q is combined by `mode` ("max" or
    "sum", saturating at the qbits ceiling). Lanes with distinct slot_ids are
    concatenated. All packets must share inventory_digest, qbits and version
    (the wire's compatibility invariants — mixing is a caller bug, not a
    merge policy).
    """
    if not packets:
        raise ValueError("merge_packets needs at least one packet")
    p0 = packets[0]
    for p in packets[1:]:
        if (p.inventory_digest != p0.inventory_digest
                or p.qbits != p0.qbits or p.version != p0.version):
            raise ValueError("merge_packets: incompatible packets "
                             "(digest/qbits/version must match)")
    qmax = (1 << p0.qbits) - 1 if p0.qbits else 0
    by_slot: dict[int, dict[int, int]] = {}
    for p in packets:
        for ln in p.lanes:
            slot = by_slot.setdefault(ln.slot_id, {})
            for cid, qv in zip(ln.concept_ids, ln.q):
                if cid in slot:
                    slot[cid] = (max(slot[cid], qv) if mode == "max"
                                 else min(slot[cid] + qv, qmax))
                else:
                    slot[cid] = qv
    lanes = tuple(Lane(sid,
                       tuple(sorted(ids)),
                       tuple(ids[c] for c in sorted(ids)))
                  for sid, ids in sorted(by_slot.items()))
    return Packet(p0.inventory_digest, p0.qbits, lanes, p0.version)
