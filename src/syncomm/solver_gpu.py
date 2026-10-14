"""GPU-vectorized batched nonnegative elastic-net extraction (FISTA).

Batched replacement for the sklearn per-lane loop behind solver.extract:
on the writer_batched workload the per-lane coordinate-descent solve costs
~18 ms x 867 lanes ~ 80% of writer wall time (measured,
artifacts/equiv/llama_t2_equiv.json). extract_fista solves the SAME per-lane
problem (solver.py module docstring; gates/j0_instrument.lock.json J0B_solver)

    min_{a >= 0}  J(a) = 0.5 * ||h - D^T a||_2^2
                       + lam1 * ||a||_1 + 0.5 * lam2 * ||a||_2^2

for all M lanes of H [M, d] simultaneously against one dictionary D [C, d]
(canonical [num_concepts, hidden] orientation, packet.py; silent transposes
forbidden), on CUDA when available and on CPU otherwise -- one code path.

Method
------
Split J = f + g with f(a) = 0.5*||h - D^T a||^2 + 0.5*lam2*||a||^2 (smooth)
and g(a) = lam1*||a||_1 + indicator(a >= 0). On the nonnegative orthant
||a||_1 = sum(a), so the proximal step is a single shifted ReLU:

    prox_{s*g}(z) = max(0, z - s*lam1).

With the Gram matrix G = D D^T [C, C] and correlations b = D h (per lane),

    grad f(a) = G a + lam2*a - b
    J(a) = 0.5*||h||^2 - a.b + 0.5*a.(G a) + lam1*sum(a) + 0.5*lam2*||a||^2

so after the one-time G / B = H D^T precompute, every FISTA iteration is
O(M*C^2), independent of hidden size d. Step size 1/L with
L = 1.02 * ||D D^T||_2 + lam2, where ||D D^T||_2 is estimated by 200
deterministic power iterations from the constant unit vector (G is PSD, so
||G v|| -> lambda_max from below; the 1.02 factor covers the residual
underestimate). Momentum uses per-lane adaptive restart (O'Donoghue & Candes
function scheme): any lane whose objective increases has its momentum reset
(t = 1, beta = 0) at that iteration. Iterations stop -- checked every
`check_every` iterations to avoid per-iteration device syncs -- when every
lane satisfies max KKT residual <= kkt_tol with EXACTLY solver.py's KKT
definition (r = h - D^T a, g = D r, active = a > ACTIVE_TOL):

    active   j:  | -g_j + lam1 + lam2*a_j |
    inactive j:  max(0, g_j - lam1)

Exactly-zero lanes stay exactly zero through every iteration (their gradient
is 0 and prox(-s*lam1) = 0) and through the polish below.

Active-set polish (exact, guarded; polish=True default)
-------------------------------------------------------
Stopping precisely at KKT <= 1e-3 can leave an objective gap above the
1e-5*(1+|J|) agreement bar vs the fp64 sklearn solve in the high-coherence
regime (measured 1.15e-5 at dictionary mean-share 0.75: along the shared
mode the problem is only lam2-strongly-convex, so a small KKT residual does
not force a small objective gap). The FISTA support is therefore refined to
the EXACT optimum by a batched active-set loop (Lawson-Hanson / FNNLS style,
seeded by the FISTA active set S = {j : a_j > ACTIVE_TOL}):

    solve (D_S D_S^T + lam2 I) a_S = D_S h - lam1  batched over lanes
      -> drop all negative coordinates from S and re-solve;
      -> if feasible, add the worst inactive violator (g_j - lam1 > tol);
    until no drops or adds remain (<= _POLISH_ROUNDS, observed <= 2).

The restricted stationary point with a_S > 0 and no inactive violation IS
the global optimum (KKT sufficiency for the convex J). Per-lane guarded
acceptance: a lane takes the polished solution only if its recomputed KKT
residual does not exceed the FISTA iterate's; otherwise it keeps the FISTA
iterate, which already meets the kkt_tol contract. Measured effect: KKT
1e-3 -> ~1e-13 and objective agreement with solver.extract to <1e-15
relative on every tested regime. polish=False gives the raw FISTA stop.

Numerics and determinism
------------------------
Internal solve in float64 on the solve device (matching solver.py's fp64
internal solve; kkt_tol is ABSOLUTE, so fp32 iterations would stall near
~1e-4 on residual-scale lanes). Final reported objective / KKT residuals
are recomputed from the explicit residual r = h - D^T a, not the Gram form,
so they are numerically the same quantities solver.py reports. No
randomness anywhere; fixed iteration schedule and power-iteration start:
identical inputs on the same device give bitwise-identical coefficients
(unit-tested on CPU in tests/test_solver_gpu.py).

Benchmark (writes artifacts/equiv/solver_gpu_bench.json; run on the node):

    PYTHONPATH=src OMP_NUM_THREADS=8 python3 -m syncomm.solver_gpu \
        --lanes 867 --also-cpu

NOT yet wired into writer_batched.py -- that swap is a separate change.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import os
import time
from dataclasses import dataclass

import torch

from .solver import ACTIVE_TOL, DEFAULT_LAM1, DEFAULT_LAM2, _check_orientation

MAX_ITER = 2000          # fixed FISTA iteration budget (task spec)
KKT_TOL = 1e-3           # convergence: max per-lane KKT residual (J0-B gate value)
CHECK_EVERY = 10         # host-side convergence checks every N iterations
POWER_ITERS = 200        # deterministic power iterations for ||D D^T||_2
LIPSCHITZ_SAFETY = 1.02  # multiplicative guard on the spectral-norm estimate

# (D-identity, lam2) -> (D strong ref, Gram, Lipschitz). See extract_fista.
_PRECOMP_CACHE: dict = {}
_POLISH_ROUNDS = 30      # cap on active-set add/drop rounds (observed <= 2)
_ADD_TOL = 1e-10         # inactive-violation slack before adding a coordinate


@dataclass
class BatchSolveResult:
    """Batched analog of solver.SolveResult: one solve, per-lane tensors."""
    a: torch.Tensor                # [M, C] coefficients ([C] for 1-D input h),
                                   #   input dtype, on the SOLVE device
    kkt: torch.Tensor              # [M] fp64 CPU per-lane KKT residuals
    objective: torch.Tensor        # [M] fp64 CPU per-lane J(a), residual form
    active_set_size: torch.Tensor  # [M] int64 CPU |{j : a_j > ACTIVE_TOL}|
    n_iter: int                    # FISTA iterations executed (shared, lanes
                                   #   iterate together until ALL converge)
    n_restarts: int                # total per-lane momentum restarts
    converged: bool                # all lanes kkt <= kkt_tol
    polished: int                  # lanes whose polished solution was accepted
    wall_time_s: float             # end-to-end (transfers + solve + diagnostics)
    lipschitz: float               # L = LIPSCHITZ_SAFETY*||D D^T||_2 + lam2
    device: str                    # solve device


def _spectral_norm_psd(G: torch.Tensor, iters: int = POWER_ITERS) -> float:
    """||G||_2 for PSD G by fixed-count power iteration. Deterministic:
    constant unit start vector, no data-dependent branching, one device sync
    (the final float())."""
    C = G.shape[0]
    v = torch.full((C,), 1.0 / math.sqrt(C), dtype=G.dtype, device=G.device)
    lam = torch.zeros((), dtype=G.dtype, device=G.device)
    for _ in range(iters):
        w = G @ v
        lam = torch.linalg.vector_norm(w)
        v = w / lam.clamp_min(1e-300)
    return float(lam)


def _kkt_and_objective(A, H64, D64, lam1: float, lam2: float):
    """solver.py's declared diagnostics, batched, explicit-residual form."""
    R = H64 - A @ D64                       # [M, d]
    g = R @ D64.T                           # [M, C]  (= D r per lane)
    act = A > ACTIVE_TOL
    kkt = torch.where(act, (lam1 + lam2 * A - g).abs(),
                      torch.clamp_min(g - lam1, 0.0)).amax(1)
    obj = (0.5 * (R * R).sum(1) + lam1 * A.sum(1)
           + 0.5 * lam2 * (A * A).sum(1))
    return kkt, obj


def _restricted_solve(S, G, B, lam1: float, lam2: float):
    """Batched exact solve of (G_SS + lam2 I) x_S = b_S - lam1 over per-lane
    supports S [M, C] bool. Supports are gathered into padded [M, k_max]
    index blocks; pad rows are identity rows with rhs 0, so pad solutions
    are exactly 0 and decouple from the real block. Returns (x [M, k_max],
    mask [M, k_max] real-coordinate flags, idx [M, k_max] gathered coord
    ids, a_cand [M, C] with negative coordinates clamped for candidate
    assembly; x itself is unclamped so the caller can detect infeasibility).
    k_max == 0 returns empty x/mask/idx and a_cand = 0."""
    M, C = S.shape
    k_max = int(S.sum(1).max())
    if k_max == 0:
        empty = B.new_zeros(M, 0)
        return (empty, empty.bool(), empty.to(torch.int64),
                B.new_zeros(M, C))
    order = torch.argsort(S.to(torch.int64), dim=1, descending=True,
                          stable=True)
    idx = order[:, :k_max]                              # [M, k] support first,
    mask = S.gather(1, idx)                             #   ascending ids
    Gsub = G[idx.unsqueeze(2), idx.unsqueeze(1)]        # [M, k, k]
    pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    Gsub = torch.where(pair, Gsub, Gsub.new_zeros(()))
    Gsub = Gsub + torch.diag_embed(mask.to(G.dtype) * lam2
                                   + (~mask).to(G.dtype))
    rhs = torch.where(mask, B.gather(1, idx) - lam1, B.new_zeros(()))
    x = torch.linalg.solve(Gsub, rhs.unsqueeze(2)).squeeze(2)
    x = x * mask
    # Scatter back to [M, C]; pad indices are redirected to a dummy column
    # C so scatter never sees duplicate REAL indices (deterministic).
    ext = B.new_zeros(M, C + 1)
    idx_ext = torch.where(mask, idx, torch.full_like(idx, C))
    ext.scatter_(1, idx_ext, torch.clamp_min(x, 0.0))
    return x, mask, idx, ext[:, :C]


def _polish(A, G, B, lam1: float, lam2: float):
    """Batched active-set refinement seeded by the FISTA support (module
    docstring). Returns (A_pol [M, C], rounds). Lanes that never reach a
    feasible restricted solution keep their FISTA coefficients in A_pol;
    the caller's KKT-guarded acceptance handles the rest."""
    M, C = A.shape
    S = A > ACTIVE_TOL
    A_pol = A.clone()
    alive = torch.ones(M, dtype=torch.bool, device=A.device)
    rounds = 0
    for _ in range(_POLISH_ROUNDS):
        if not bool(alive.any()):
            break
        rounds += 1
        try:
            x, mask, idx, a_cand = _restricted_solve(S, G, B, lam1, lam2)
        except RuntimeError:            # singular restricted system (only
            break                       #   possible at lam2 == 0): keep FISTA
        if x.shape[1] == 0:
            has_neg = torch.zeros(M, dtype=torch.bool, device=A.device)
        else:
            neg = (x < 0) & mask
            has_neg = neg.any(1)
            if bool(has_neg.any()):     # drop ALL negative coords (FNNLS)
                negfull = torch.zeros(M, C + 1, dtype=torch.bool,
                                      device=A.device)
                negfull.scatter_(1, torch.where(mask, idx,
                                                torch.full_like(idx, C)), neg)
                S = S & ~negfull[:, :C]
        feasible = alive & ~has_neg
        A_pol = torch.where(feasible.unsqueeze(1), a_cand, A_pol)
        # worst inactive violator per feasible lane: g_j - lam1 > _ADD_TOL
        g_full = B - a_cand @ G
        viol = (g_full - lam1).masked_fill(S, float("-inf"))
        vstar, jstar = viol.max(1)
        needs_add = feasible & (vstar > _ADD_TOL)
        if bool(needs_add.any()):
            addhot = torch.zeros(M, C, dtype=torch.bool, device=A.device)
            addhot.scatter_(1, jstar.unsqueeze(1), needs_add.unsqueeze(1))
            S = S | addhot
        alive = (alive & has_neg) | needs_add
    return A_pol, rounds


@torch.no_grad()
def extract_fista(H: torch.Tensor, D: torch.Tensor,
                  lam1: float = DEFAULT_LAM1, lam2: float = DEFAULT_LAM2,
                  device=None, *, max_iter: int = MAX_ITER,
                  kkt_tol: float = KKT_TOL, check_every: int = CHECK_EVERY,
                  polish: bool = True) -> BatchSolveResult:
    """Batched nonnegative elastic-net extraction a_m = Phi(h_m; D) for all
    lanes simultaneously (module docstring for the method).

    Args:
      H: [M, d] stack of lanes, or [d] a single lane.
      D: [C, d] dictionary, canonical [num_concepts, hidden] orientation.
      lam1, lam2: penalties of the UNSCALED objective (solver.py values).
      device: solve device; None -> "cuda" if available else "cpu".
      max_iter: fixed FISTA iteration budget.
      kkt_tol: ABSOLUTE convergence bar on the per-lane KKT residual
        (solver.py definition); iteration stops when ALL lanes are below it.
      check_every: iterations between host-side convergence checks.
      polish: run the guarded exact active-set refinement (recommended).

    Returns:
      BatchSolveResult. `a` is [M, C] in H's dtype on the SOLVE device
      ([C] for 1-D input); per-lane diagnostics are fp64 CPU tensors and are
      computed at the internal float64 solution, explicit-residual form --
      the same quantities solver.extract reports.

    Deterministic for fixed inputs on a fixed device; identical to a fresh
    process rerun (no global state, no randomness).
    """
    if lam1 < 0 or lam2 < 0:
        raise ValueError(
            f"penalties must be nonnegative, got lam1={lam1} lam2={lam2}")
    if lam1 + lam2 <= 0:
        raise ValueError("lam1 + lam2 must be > 0 (unpenalized NNLS is out "
                         "of scope for the J0-B instrument)")
    _check_orientation(H, D)
    t0 = time.perf_counter()
    dev = (torch.device(device) if device is not None
           else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    single = H.dim() == 1
    out_dtype = H.dtype
    H64 = (H.detach().reshape(1, -1) if single else H.detach())
    H64 = H64.to(dev, torch.float64)
    D64 = D.detach().to(dev, torch.float64)
    M, _ = H64.shape
    C = D64.shape[0]

    # One-time precompute: every iteration below is O(M*C^2), d-free.
    # G and L depend ONLY on (D, lam2), yet the write path re-installs a hook
    # per band layer per (item, arm, seed) — up to 36 recomputes of a
    # [C,C] fp64 Gram (C=20431 -> 1.7 TFLOP) per item, which is what made the
    # 2026-08-05 campaign run at 2.4 min/item. Caching is BIT-IDENTICAL by
    # construction: the matmul is deterministic for fixed input/shape/device
    # and _spectral_norm_psd is a fixed-count power iteration from a constant
    # start vector. Off by default; SYNCOMM_SOLVER_CACHE=1 enables it, so a
    # running campaign never changes behaviour without an explicit opt-in.
    _ck = None
    if os.environ.get("SYNCOMM_SOLVER_CACHE") == "1":
        _ck = (D.data_ptr(), tuple(D.shape), str(D.dtype), str(dev), float(lam2))
        _hit = _PRECOMP_CACHE.get(_ck)
    else:
        _hit = None
    if _hit is not None:
        _D_ref, G, L = _hit          # _D_ref pins D so data_ptr cannot recycle
    else:
        G = D64 @ D64.T                                 # [C, C]
        L = LIPSCHITZ_SAFETY * _spectral_norm_psd(G) + lam2
        if _ck is not None:
            _PRECOMP_CACHE[_ck] = (D, G, L)
    B = H64 @ D64.T                                     # [M, C]
    hh = (H64 * H64).sum(1)                             # [M]
    step = 1.0 / max(L, 1e-12)

    A = torch.zeros(M, C, dtype=torch.float64, device=dev)
    Y = torch.zeros_like(A)
    t = torch.ones(M, dtype=torch.float64, device=dev)
    J_prev = 0.5 * hh.clone()                           # J(0) per lane
    restarts = torch.zeros((), dtype=torch.int64, device=dev)
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        grad = Y @ G + lam2 * Y - B
        A_new = torch.clamp_min(Y - step * (grad + lam1), 0.0)  # prox step
        AG = A_new @ G
        J_new = (0.5 * hh - (A_new * B).sum(1) + 0.5 * (A_new * AG).sum(1)
                 + lam1 * A_new.sum(1) + 0.5 * lam2 * (A_new * A_new).sum(1))
        g = B - AG                                      # = D r, Gram form
        act = A_new > ACTIVE_TOL
        kkt_it = torch.where(act, (lam1 + lam2 * A_new - g).abs(),
                             torch.clamp_min(g - lam1, 0.0)).amax(1)
        rs = J_new > J_prev                             # per-lane restart
        restarts += rs.sum()
        t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        beta = torch.where(rs, torch.zeros_like(t), (t - 1.0) / t_next)
        t = torch.where(rs, torch.ones_like(t), t_next)
        Y = A_new + beta.unsqueeze(1) * (A_new - A)
        A = A_new
        J_prev = J_new
        if ((it % check_every == 0 or it == max_iter)
                and bool((kkt_it <= kkt_tol).all())):
            break

    n_polished = 0
    kkt_f, obj_f = _kkt_and_objective(A, H64, D64, lam1, lam2)
    A_final = A
    if polish:
        A_pol, _ = _polish(A, G, B, lam1, lam2)
        kkt_p, obj_p = _kkt_and_objective(A_pol, H64, D64, lam1, lam2)
        accept = kkt_p <= kkt_f          # guarded per-lane acceptance
        A_final = torch.where(accept.unsqueeze(1), A_pol, A)
        kkt_f = torch.where(accept, kkt_p, kkt_f)
        obj_f = torch.where(accept, obj_p, obj_f)
        n_polished = int(accept.sum())

    a_out = A_final.to(out_dtype)
    if single:
        a_out = a_out.squeeze(0)
    result = BatchSolveResult(
        a=a_out,
        kkt=kkt_f.to("cpu"),
        objective=obj_f.to("cpu"),
        active_set_size=(A_final > ACTIVE_TOL).sum(1).to("cpu"),
        n_iter=n_iter,
        n_restarts=int(restarts),
        converged=bool((kkt_f <= kkt_tol).all()),
        polished=n_polished,
        wall_time_s=0.0,                 # set below, after all syncs
        lipschitz=float(L),
        device=str(dev),
    )
    result.wall_time_s = time.perf_counter() - t0
    return result


# --------------------------------------------------------------------------
# Real-scale benchmark: sklearn per-lane loop (solver.extract, the deployed
# writer_batched path) vs extract_fista on the llama-3.1-8b jlens2
# dictionary [221, 4096], 867 lanes -- the exact workload of
# artifacts/equiv/llama_t2_equiv.json. Lane recipe calibrated on the dev box
# so the sklearn side reproduces the measured ~18 ms/lane bottleneck.
# --------------------------------------------------------------------------

def _bench_lanes(D: torch.Tensor, mean_pullback, n_lanes: int, n_mean: int,
                 seed: int) -> torch.Tensor:
    """Deterministic realistic lane mix:
      - concept-bearing: h = a_true @ D + noise, k=5 support, amps U(5, 15),
        ||noise|| = 5 (dev-box calibrated: sklearn ~18.5 ms/lane, 20-35
        active -- matches the measured equiv-run regime);
      - non-concept: random directions at residual scale 15 (~12 ms/lane);
      - mean-h: the dictionary's own mean_pullback vector on a dose ladder
        1..30 (spans the writer's deployed amp ~9.5).
    """
    g = torch.Generator().manual_seed(seed)
    C, d = D.shape
    if mean_pullback is None:
        n_mean = 0
    n_combo = (n_lanes - n_mean) // 2
    n_rand = n_lanes - n_mean - n_combo
    lanes = []
    for _ in range(n_combo):
        supp = torch.randperm(C, generator=g)[:5]
        a = torch.zeros(C)
        a[supp] = 10.0 * (0.5 + torch.rand(5, generator=g))
        noise = torch.randn(d, generator=g)
        lanes.append(a @ D + noise / noise.norm() * 5.0)
    R = torch.randn(n_rand, d, generator=g)
    lanes.extend(R / R.norm(dim=1, keepdim=True) * 15.0)
    if n_mean > 0:
        mp = mean_pullback.reshape(-1).float()
        lanes.extend(s * mp for s in torch.linspace(1.0, 30.0, n_mean))
    return torch.stack([l.float() for l in lanes])


def _cosine_rows(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    nx, ny = X.norm(dim=1), Y.norm(dim=1)
    num = (X * Y).sum(1)
    both_zero = (nx == 0) & (ny == 0)
    cos = num / (nx * ny).clamp_min(1e-300)
    return torch.where(both_zero, torch.ones_like(cos), cos)


def _bench(args) -> None:
    from . import solver as solver_cpu

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    dict_path = args.dict or os.path.join(
        repo, "artifacts", "jlens2", "llama-3.1-8b_tap0.5_inv2.pt")
    out_path = args.out or os.path.join(
        repo, "artifacts", "equiv", "solver_gpu_bench.json")

    blob = torch.load(dict_path, map_location="cpu", weights_only=False)
    D = blob[args.field].float().contiguous()
    mp = blob.get("mean_pullback")
    mp = mp if isinstance(mp, torch.Tensor) else None
    C, d = D.shape
    n_mean_eff = args.mean_lanes if mp is not None else 0
    H = _bench_lanes(D, mp, args.lanes, args.mean_lanes, args.seed)
    M = H.shape[0]
    print(f"[bench] dict={os.path.basename(dict_path)} field={args.field} "
          f"C={C} d={d} lanes={M} (mean-h lanes: {n_mean_eff})")

    # ---- sklearn per-lane loop (deployed path: ONE extract call on [M, d])
    t0 = time.perf_counter()
    res_sk = solver_cpu.extract(H, D)
    t_sk = time.perf_counter() - t0
    A_sk = torch.stack([r.a for r in res_sk]).to(torch.float64)
    obj_sk = torch.tensor([r.objective for r in res_sk], dtype=torch.float64)
    kkt_sk = torch.tensor([r.kkt_residual for r in res_sk],
                          dtype=torch.float64)
    solve_only_ms = 1e3 * sum(r.wall_time_s for r in res_sk) / M
    print(f"[bench] sklearn loop: {t_sk:.2f}s total, "
          f"{1e3 * t_sk / M:.2f} ms/lane ({solve_only_ms:.2f} ms/lane in "
          f"per-lane solves), kkt_max={float(kkt_sk.max()):.2e}, "
          f"converged={all(r.converged for r in res_sk)}")

    # ---- extract_fista (end-to-end timed, transfers included)
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _ = extract_fista(H[:8], D, device=dev)      # warmup: context + kernels
    times, res_f = [], None
    for _ in range(args.reps):
        t1 = time.perf_counter()
        res_f = extract_fista(H, D, device=dev)
        times.append(time.perf_counter() - t1)
    t_f = sorted(times)[len(times) // 2]
    print(f"[bench] extract_fista[{res_f.device}]: reps="
          f"{[round(x, 4) for x in times]} median {t_f:.4f}s "
          f"({1e3 * t_f / M:.3f} ms/lane), n_iter={res_f.n_iter}, "
          f"restarts={res_f.n_restarts}, polished={res_f.polished}/{M}, "
          f"kkt_max={float(res_f.kkt.max()):.2e}, "
          f"converged={res_f.converged}")

    fista_cpu = None
    if args.also_cpu:
        t1 = time.perf_counter()
        res_c = extract_fista(H, D, device="cpu")
        fista_cpu = {"wall_s": time.perf_counter() - t1,
                     "n_iter": res_c.n_iter,
                     "kkt_max": float(res_c.kkt.max()),
                     "converged": res_c.converged}
        print(f"[bench] extract_fista[cpu]: {fista_cpu['wall_s']:.2f}s")

    # ---- per-lane agreement
    A_f = res_f.a.detach().to("cpu", torch.float64)
    cos = _cosine_rows(A_sk, A_f)
    dJ = (res_f.objective - obj_sk).abs() / (1.0 + obj_sk.abs())
    s_sk, s_f = A_sk > ACTIVE_TOL, A_f > ACTIVE_TOL
    union = (s_sk | s_f).sum(1)
    jac = torch.where(union > 0,
                      (s_sk & s_f).sum(1).double() /
                      union.double().clamp_min(1.0),
                      torch.ones(M, dtype=torch.float64))
    speedup = t_sk / t_f
    report = {
        "ts_utc": time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "torch": torch.__version__,
        "device": str(res_f.device),
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "cpu"),
        "dict": os.path.abspath(dict_path),
        "field": args.field,
        "C": C, "d": d, "n_lanes": M,
        "lane_recipe": {
            "seed": args.seed,
            "concept_bearing": {"n": (M - n_mean_eff) // 2, "k": 5,
                                "amp": "U(5,15)", "noise_norm": 5.0},
            "random_dirs": {"n": M - n_mean_eff - (M - n_mean_eff) // 2,
                            "scale": 15.0},
            "mean_h": {"n": n_mean_eff, "dose_ladder": "linspace(1,30)"},
        },
        "sklearn": {
            "wall_total_s": t_sk,
            "per_lane_ms": 1e3 * t_sk / M,
            "per_lane_solve_only_ms": solve_only_ms,
            "kkt_max": float(kkt_sk.max()),
            "all_converged": all(r.converged for r in res_sk),
        },
        "fista": {
            "device": str(res_f.device),
            "wall_s_reps": times,
            "wall_s_median": t_f,
            "per_lane_ms": 1e3 * t_f / M,
            "n_iter": res_f.n_iter,
            "n_restarts": res_f.n_restarts,
            "lipschitz": res_f.lipschitz,
            "polished_lanes": res_f.polished,
            "kkt_max": float(res_f.kkt.max()),
            "converged": res_f.converged,
        },
        "fista_cpu": fista_cpu,
        "agreement": {
            "cosine_min": float(cos.min()),
            "cosine_median": float(cos.median()),
            "obj_rel_diff_max": float(dJ.max()),
            "obj_rel_diff_median": float(dJ.median()),
            "support_jaccard_min": float(jac.min()),
            "support_jaccard_median": float(jac.median()),
            "active_mean_sklearn": float(s_sk.sum(1).double().mean()),
            "active_mean_fista": float(s_f.sum(1).double().mean()),
        },
        "speedup_vs_sklearn": speedup,
        "target_speedup_50x_pass": bool(speedup >= 50.0),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[bench] agreement: cos_min={report['agreement']['cosine_min']:.9f}"
          f" dJ_rel_max={report['agreement']['obj_rel_diff_max']:.2e} "
          f"jaccard_min={report['agreement']['support_jaccard_min']:.3f}")
    print(f"[bench] SPEEDUP {speedup:.1f}x (target >= 50x: "
          f"{'PASS' if speedup >= 50 else 'FAIL'}) -> {out_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=(
        "Benchmark solver.extract (sklearn per-lane loop) vs extract_fista "
        "on a real jlens2 dictionary; writes a JSON report."))
    p.add_argument("--dict", default=None,
                   help="dictionary .pt (default: llama-3.1-8b_tap0.5_inv2)")
    p.add_argument("--field", default="dirs")
    p.add_argument("--lanes", type=int, default=867)
    p.add_argument("--mean-lanes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--also-cpu", action="store_true")
    p.add_argument("--out", default=None)
    _bench(p.parse_args(argv))


if __name__ == "__main__":
    main()
