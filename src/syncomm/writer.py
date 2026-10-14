"""Slot writers (refocus 02 §8): position-targeted writes at one decoder layer.

Modes:
  add      h[p] += vec                     (carrier-slot addition, §8.2)
  replace  h[p] += D_S^T (a_hat_S - a0_S)  (slot-wise sparse-coefficient
           replacement, §8.3: remove the receiver's current realization of the
           SELECTED coordinates, install the communicated one; the complement
           of the residual is untouched)

Both fire ONLY on prefill chunks (T > 1); cached decode steps are untouched —
the written state conditions generation through ordinary KV attention. This is
the structural opposite of e015.core.Steer (all positions, every chunk), which
remains the global-addition diagnostic baseline.

Positions are indices into the UNPADDED prompt per batch row; left padding
(core.load sets padding_side="left") is corrected per row via the attention
mask. a0 is computed at write time by the declared extractor Phi (solver.extract
on the captured residual, fp32); a_hat comes from the packet, already in
receiver-native units (calibration is the caller's responsibility).

Zero-write guarantee: vec=None / empty support / a_hat == a0 leaves the forward
numerically identical (unit-tested).
"""
from __future__ import annotations

import torch

from .cost_timing import sync as _cost_sync, perf as _cost_perf   # cost axis (2026-09-12)
from .e015.core import decoder_layers
from . import solver as _solver

try:                                  # 105.9x batched FISTA (solver_gpu bench)
    from . import solver_gpu as _solver_gpu
except Exception:                     # pragma: no cover - optional accelerator
    _solver_gpu = None

# The a0 baseline solve dominated writer wall-time: one sklearn CD fit per
# (row, position) INSIDE the CUDA hook, ~12.7 ms each x ~21 per forward = 89%
# of injection forward time -- which is also why the deployed channel loses the
# latency axis outright (38.2 ms in-hook vs 0.32-0.81 ms of extra text prefill,
# WHERE_LATENT_WINS_2026-08-02 H3). All positions of one hook are now solved in
# a single batched call, on GPU when solver_gpu imports.
USE_GPU_SOLVER = _solver_gpu is not None


def _solve_a0(H, D, lam1, lam2):
    """H [M, d] -> coefficients [M, C]; batched, GPU when available.

    device=D.device pins the returned coefficients to the caller's tensors:
    extract_fista's default (cuda:0) returns a on the SOLVE device, which
    crashed the hook whenever D lived elsewhere (CPU-loaded dicts, or shards
    tapped on cuda:1+ under device_map).
    """
    if USE_GPU_SOLVER:
        return _solver_gpu.extract_fista(H, D, lam1=lam1, lam2=lam2,
                                         device=D.device).a
    return torch.stack([_solver.extract(H[i], D, lam1=lam1, lam2=lam2).a
                        for i in range(H.shape[0])])


class SlotWrite:
    """Context manager installing a prefill-only, position-masked write hook."""

    def __init__(self, model, li: int, positions: list[list[int]],
                 mode: str, *, vec: torch.Tensor | None = None,
                 D: torch.Tensor | None = None,
                 support: list[int] | None = None,
                 a_hat: torch.Tensor | None = None,
                 lam1: float = 0.05, lam2: float = 1e-3,
                 row_lengths: list[int] | None = None,
                 target_norm: dict[int, float] | None = None,
                 record_norms: dict | None = None):
        """positions[b] = token indices into row b's unpadded prompt.
        row_lengths[b] = unpadded length of row b (for left-pad offset); if
        None, rows are assumed unpadded (batch of equal-length prompts).
        mode='add' uses vec [d]; mode='replace' uses D [C, d] + support +
        a_hat [len(support)] and solves a0 per position via solver.extract."""
        if mode not in ("add", "replace"):
            raise ValueError(mode)
        if mode == "add" and vec is None:
            raise ValueError("add mode needs vec")
        if mode == "replace" and (D is None or support is None or a_hat is None):
            raise ValueError("replace mode needs D, support, a_hat")
        self.model, self.li, self.mode = model, li, mode
        self.positions, self.row_lengths = positions, row_lengths
        self.vec, self.D, self.support = vec, D, support
        self.a_hat = a_hat
        self.lam1, self.lam2 = lam1, lam2
        self.n_writes = 0
        # 2026-09-13 (external review): row-normalized dictionaries do NOT give a matched write --
        # ||delta||^2 = (a_hat-a0)^T (D D^T) (a_hat-a0) depends on the angles between the written
        # rows and on a0, both of which change with the dictionary. Measured, the Gaussian control
        # wrote 0.64-0.85 of MC's norm. target_norm maps a write POSITION to the norm the delta at
        # that position must have, so a control can be matched per site instead of per row. None
        # (the default) leaves every existing path byte-identical.
        self.target_norm = target_norm
        # record_norms collects the norm this hook ACTUALLY writes at each (layer, position). The
        # norms measured outside the forward (delta_stats, measure_sites) are exact only at the
        # lowest written layer, because a0 above it is read off the un-intervened forward; matching
        # a control against those proxies inflates the upper layers (measured: factors up to 4x).
        # A control is therefore matched against what the reference arm really wrote, recorded here.
        self.record_norms = record_norms
        self.n_rescaled = 0
        self.applied_norms: list[float] = []
        self.rescale_factors: list[float] = []
        self.t_solve_s = 0.0            # in-hook a0 solve seconds (cost axis, 2026-09-12; additive)
        self.handle = decoder_layers(model)[li].register_forward_hook(self._hk)

    def _hk(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        B, T, d = h.shape
        if T == 1:                       # cached decode step: never touch
            return out
        h = h.clone()
        sites = []
        for b in range(min(B, len(self.positions))):
            off = 0 if self.row_lengths is None else T - self.row_lengths[b]
            for p in self.positions[b]:
                idx = off + p
                if 0 <= idx < T:
                    sites.append((b, idx))
        if self.mode == "add":
            for b, idx in sites:
                h[b, idx] = h[b, idx] + self.vec.to(h.dtype).to(h.device)
            self.n_writes += len(sites)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        if sites:
            Dw = self.D.float()
            if not USE_GPU_SOLVER:
                Dw = Dw.cpu()
            elif h.is_cuda:
                Dw = Dw.to(h.device)   # solve where the hook fires
            H = torch.stack([h[b, idx].detach().float().to(Dw.device)
                             for b, idx in sites])          # [M, d]
            _cost_sync(H)
            _t0 = _cost_perf()
            A0 = _solve_a0(H, Dw, self.lam1, self.lam2)      # [M, C]
            _cost_sync(A0)
            self.t_solve_s += _cost_perf() - _t0
            Ds = Dw[self.support]                            # [k, d]
            ah = self.a_hat.float().to(Dw.device)
            # Batch only the SOLVE (89% of writer cost); apply each lane's
            # delta with the same per-lane GEMV the reference path uses, so
            # batching cannot change a single bit of the written state
            # (a batched GEMM here reorders the reduction and shifts 1 fp32
            # ULP -- measured 1.19e-07 on 12/240 elements, 2026-08-02).
            # NOTE (2026-09-07): this is a sparse SYNTHESIS edit, not coordinate replacement.
            # It sets the addressed rows' contribution to h, and does NOT guarantee that
            # re-encoding h' returns a_hat on the support or leaves off-support coefficients
            # fixed -- on an overcomplete, non-orthogonal dictionary it does neither. Worked
            # counterexample: D two identical unit rows, h=1, lam1=0.05, lam2=1e-3 encodes to
            # (0.47476, 0.47476); asking to zero coordinate 0 gives h'=0.52524, which re-encodes
            # to (0.23750, 0.23750) -- the target is not zeroed and the other coordinate moved.
            # The causal results do not depend on the replacement reading; the estimand is the
            # behavioural contrast, not a claim about the receiver's own decomposition.
            for j, (b, idx) in enumerate(sites):
                delta = (ah - A0[j, self.support].float()) @ Ds
                if self.target_norm is not None:
                    off = 0 if self.row_lengths is None else T - self.row_lengths[b]
                    tgt = self.target_norm.get(int(idx - off))
                    if tgt is not None:
                        cur = float(delta.norm())
                        if cur > 0.0:
                            f = float(tgt) / cur
                            delta = delta * f
                            self.n_rescaled += 1
                            self.rescale_factors.append(f)
                _applied_nrm = float(delta.norm())
                self.applied_norms.append(_applied_nrm)
                if self.record_norms is not None:
                    off0 = 0 if self.row_lengths is None else T - self.row_lengths[b]
                    self.record_norms[(int(self.li), int(idx - off0))] = _applied_nrm
                h[b, idx] = h[b, idx] + delta.to(h.dtype).to(h.device)
            self.n_writes += len(sites)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
