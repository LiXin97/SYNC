"""J0-B nonnegative sparse extraction solver (Phi).

Solves, per message lane (research spec 02 §5; gates/j0_instrument.lock.json
J0B_solver; ADR-0003 D5 scaffolding item):

    min_{a >= 0}  J(a) = 0.5 * ||h - D^T a||_2^2
                       + lam1 * ||a||_1
                       + 0.5 * lam2 * ||a||_2^2

with h in R^d a lane activation, D in R^{C x d} the concept dictionary in the
canonical [num_concepts, hidden] orientation (packet.py / the project rules §8; silent
transposes forbidden), and a in R^C_{>=0} the coefficient vector. Defaults
lam1=0.05, lam2=1e-3 are J-CoT's published values (the project rules §2 known facts),
re-confirmed by the re-extraction threshold ladder in tests/test_solver.py.

Backend: sklearn.linear_model.ElasticNet(positive=True, fit_intercept=False).

Exact (lam1, lam2) -> (alpha, l1_ratio) mapping
-----------------------------------------------
sklearn (verified against the 1.7.0 docstring) minimizes

    1/(2*n_samples) * ||y - X w||_2^2
      + alpha * l1_ratio * ||w||_1
      + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2

With X = D^T (shape [d, C], so n_samples = d), y = h, w = a, divide our
objective J(a) by d (a positive constant, so the argmin is unchanged):

    J(a)/d = 1/(2d) * ||h - D^T a||^2 + (lam1/d) * ||a||_1
             + 0.5 * (lam2/d) * ||a||^2

Matching term by term:

    alpha * l1_ratio       = lam1 / d
    alpha * (1 - l1_ratio) = lam2 / d
    =>  alpha    = (lam1 + lam2) / d
        l1_ratio = lam1 / (lam1 + lam2)

The 1/d factor is exactly sklearn's 1/(2*n_samples) convention; omitting it
would over-regularize by a factor of d = hidden size. The mapping is verified
numerically in tests/test_solver.py two independent ways: (i) the KKT
stationarity residual of the UNSCALED objective J at the sklearn solution is
driven to ~0 (a wrong 1/n factor would leave a residual of order d*lam1), and
(ii) coefficient agreement (cosine >= 0.99) with `reference_solve`, an
L-BFGS-B minimizer of J itself.

KKT conditions of J at a* (used for the declared-stationarity diagnostic),
with r = h - D^T a* the residual and g = D r the correlation vector:

    active   (a_j > 0):  -g_j + lam1 + lam2 * a_j  = 0
    inactive (a_j = 0):  -g_j + lam1              >= 0   i.e.  g_j <= lam1

kkt_residual = max( max_{active j} | -g_j + lam1 + lam2*a_j | ,
                    max_{inactive j} max(0, g_j - lam1) )        (0 if a set
is empty). Gate J0-B requires median kkt_residual <= 1e-3 on the locked
instrument set; the coordinate-descent tolerance below leaves large margin.

Determinism: fixed max_iter/tol, selection='cyclic', float64 internal solve,
no randomness anywhere -> identical inputs give bitwise-identical
coefficients (unit-tested).
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

# J-CoT published values (the project rules §2). DEFAULT_LAM1 = 0.05 passes the frozen
# J0-B re-extraction thresholds (median rel err <= 0.20, p90 <= 0.40) on
# well-conditioned synthetic problems -- see tests/test_solver.py ladder.
DEFAULT_LAM1 = 0.05
DEFAULT_LAM2 = 1e-3

# Coefficients below this are "inactive". sklearn coordinate descent returns
# exact zeros for inactive coordinates, so this only guards float noise in
# user-supplied coefficient vectors (reextract_error).
ACTIVE_TOL = 1e-10

_MAX_ITER = 100_000
_TOL = 1e-10  # sklearn dual-gap tol (scaled by ||y||^2 / n_samples internally)


@dataclass
class SolveResult:
    """Per-solve diagnostics (research spec 02 §5 required extractor set)."""
    a: torch.Tensor        # [C] nonnegative coefficients, dtype = input h dtype
    active_set_size: int   # |{j : a_j > ACTIVE_TOL}|
    objective: float       # J(a) at the solution (unscaled objective)
    residual_norm: float   # ||h - D^T a||_2
    kkt_residual: float    # declared stationarity criterion (module docstring)
    wall_time_s: float     # wall time for this lane's solve + diagnostics
    n_iter: int            # coordinate-descent iterations (0 for the h=0 path)
    converged: bool        # no ConvergenceWarning and n_iter < max_iter


@dataclass
class ReextractResult:
    rel_err: float          # ||a2 - a|| / (||a|| + eps)
    support_jaccard: float  # |S(a) & S(a2)| / |S(a) | S(a2)|; 1.0 if both empty
    a2: torch.Tensor        # re-extracted coefficients [C]


def _check_orientation(h: torch.Tensor, D: torch.Tensor) -> None:
    if D.dim() != 2:
        raise ValueError(f"D must be 2-D [num_concepts, hidden], got {tuple(D.shape)}")
    if h.dim() not in (1, 2):
        raise ValueError(f"h must be [d] or [M, d], got {tuple(h.shape)}")
    d = D.shape[1]
    if h.shape[-1] != d:
        hint = ""
        if D.shape[0] != D.shape[1] and h.shape[-1] == D.shape[0]:
            hint = (" -- h matches D.shape[0]: D looks transposed. The canonical"
                    " orientation is [num_concepts, hidden] (packet.py); no"
                    " silent transposes.")
        raise ValueError(
            f"h last dim {h.shape[-1]} != hidden dim {d} of D {tuple(D.shape)}{hint}")


def _objective_np(a: np.ndarray, h: np.ndarray, X: np.ndarray,
                  lam1: float, lam2: float) -> float:
    """J(a) with X = D^T [d, C]; assumes a >= 0 so ||a||_1 = sum(a)."""
    r = h - X @ a
    return float(0.5 * (r @ r) + lam1 * a.sum() + 0.5 * lam2 * (a @ a))


def _kkt_residual_np(a: np.ndarray, h: np.ndarray, X: np.ndarray,
                     lam1: float, lam2: float) -> float:
    r = h - X @ a               # [d]
    g = X.T @ r                 # D r, [C]
    active = a > ACTIVE_TOL
    kkt = 0.0
    if active.any():
        kkt = float(np.max(np.abs(-g[active] + lam1 + lam2 * a[active])))
    if (~active).any():
        kkt = max(kkt, float(max(0.0, np.max(g[~active] - lam1))))
    return kkt


def _solve_single(h64: np.ndarray, X64: np.ndarray, lam1: float, lam2: float,
                  out_dtype: torch.dtype) -> SolveResult:
    t0 = time.perf_counter()
    d = X64.shape[0]

    if not np.any(h64):
        # Exact zero lane: the unique minimizer is a = 0 (any a_j > 0 strictly
        # increases J). Short-circuit -- sklearn's dual-gap stopping rule
        # degenerates when ||y|| = 0 (tol scales with ||y||^2). KKT holds
        # exactly: g = 0, inactive violation = max(0, -lam1) = 0.
        return SolveResult(
            a=torch.zeros(X64.shape[1], dtype=out_dtype),
            active_set_size=0,
            objective=0.0,
            residual_norm=0.0,
            kkt_residual=0.0,
            wall_time_s=time.perf_counter() - t0,
            n_iter=0,
            converged=True,
        )

    alpha = (lam1 + lam2) / d
    l1_ratio = lam1 / (lam1 + lam2)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, positive=True,
                       fit_intercept=False, max_iter=_MAX_ITER, tol=_TOL,
                       selection="cyclic", warm_start=False)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model.fit(X64, h64)
    hit_warning = any(issubclass(w.category, ConvergenceWarning) for w in wlist)
    n_iter = int(np.max(model.n_iter_)) if model.n_iter_ is not None else _MAX_ITER

    a64 = np.asarray(model.coef_, dtype=np.float64)
    # positive=True guarantees a >= 0; clip exact float noise defensively.
    a64 = np.maximum(a64, 0.0)
    r = h64 - X64 @ a64
    return SolveResult(
        a=torch.from_numpy(a64.copy()).to(out_dtype),
        active_set_size=int(np.sum(a64 > ACTIVE_TOL)),
        objective=_objective_np(a64, h64, X64, lam1, lam2),
        residual_norm=float(np.linalg.norm(r)),
        kkt_residual=_kkt_residual_np(a64, h64, X64, lam1, lam2),
        wall_time_s=time.perf_counter() - t0,
        n_iter=n_iter,
        converged=(not hit_warning) and n_iter < _MAX_ITER,
    )


def extract(h: torch.Tensor, D: torch.Tensor,
            lam1: float = DEFAULT_LAM1, lam2: float = DEFAULT_LAM2):
    """Nonnegative elastic-net extraction a = Phi(h; D, lam1, lam2).

    Args:
      h: [d] single lane, or [M, d] batch of lanes.
      D: [C, d] dictionary, rows unit-norm, canonical orientation
         [num_concepts, hidden] (packet.py convention).
      lam1, lam2: L1 / L2 penalties of the UNSCALED objective (module
         docstring gives the exact mapping onto sklearn's alpha/l1_ratio).

    Returns:
      SolveResult for 1-D h; list[SolveResult] (one per lane, in order) for
      2-D h. The batch path is defined as the per-lane loop, so batch and
      loop results are bitwise identical by construction (unit-tested).

    Deterministic: float64 internal solve, cyclic coordinate descent, fixed
    max_iter/tol, no randomness.
    """
    if lam1 < 0 or lam2 < 0:
        raise ValueError(f"penalties must be nonnegative, got lam1={lam1} lam2={lam2}")
    if lam1 + lam2 <= 0:
        raise ValueError("lam1 + lam2 must be > 0 (unpenalized NNLS is out of "
                         "scope for the J0-B instrument)")
    _check_orientation(h, D)

    # Internal solve in float64: X = D^T is [d, C] = [n_samples, n_features].
    # D is C-ordered [C, d], so D.T is naturally F-ordered -- coordinate
    # descent's preferred layout (no hidden copy inside sklearn).
    D64 = D.detach().to("cpu", torch.float64).numpy()
    X64 = D64.T
    out_dtype = h.dtype

    if h.dim() == 1:
        return _solve_single(h.detach().to("cpu", torch.float64).numpy(),
                             X64, lam1, lam2, out_dtype)
    H64 = h.detach().to("cpu", torch.float64).numpy()
    return [_solve_single(H64[i], X64, lam1, lam2, out_dtype)
            for i in range(H64.shape[0])]


def reference_solve(h: torch.Tensor, D: torch.Tensor,
                    lam1: float = DEFAULT_LAM1,
                    lam2: float = DEFAULT_LAM2) -> torch.Tensor:
    """Trusted small-problem reference minimizer of J (tests only).

    scipy.optimize.minimize L-BFGS-B with bounds a >= 0 and the exact
    gradient  grad J(a) = -D(h - D^T a) + lam1 + lam2 * a  (valid on the
    nonnegative orthant, where ||a||_1 = sum(a) is smooth). Run from multiple
    starts -- zeros and abs(least-squares init) -- keeping the lowest
    objective. Returns float64 coefficients [C].
    """
    if h.dim() != 1:
        raise ValueError("reference_solve takes a single lane h [d]")
    _check_orientation(h, D)
    h64 = h.detach().to("cpu", torch.float64).numpy()
    X = D.detach().to("cpu", torch.float64).numpy().T   # [d, C]
    C = X.shape[1]

    def f(a: np.ndarray) -> float:
        return _objective_np(a, h64, X, lam1, lam2)

    def grad(a: np.ndarray) -> np.ndarray:
        r = h64 - X @ a
        return -(X.T @ r) + lam1 + lam2 * a

    ls = np.linalg.lstsq(X, h64, rcond=None)[0]
    starts = [np.zeros(C), np.abs(ls)]
    best_a, best_f = None, np.inf
    for x0 in starts:
        res = minimize(f, x0, jac=grad, method="L-BFGS-B",
                       bounds=[(0.0, None)] * C,
                       options={"ftol": 1e-12, "gtol": 1e-12,
                                "maxiter": 50_000, "maxfun": 200_000})
        if res.fun < best_f:
            best_f, best_a = res.fun, res.x
    return torch.from_numpy(np.maximum(best_a, 0.0))


def reextract_error(a: torch.Tensor, D: torch.Tensor,
                    lam1: float = DEFAULT_LAM1,
                    lam2: float = DEFAULT_LAM2) -> ReextractResult:
    """Reconstruct-then-re-extract defect (gate J0-B: median rel_err <= 0.20,
    p90 <= 0.40 on the locked instrument set).

    v = D^T a (computed in D's dtype -- the deployment reconstruction path),
    a2 = extract(v, D, lam1, lam2).a, rel_err = ||a2 - a|| / (||a|| + eps),
    support Jaccard on {a > ACTIVE_TOL}.
    """
    a = torch.as_tensor(a)
    if a.dim() != 1 or a.shape[0] != D.shape[0]:
        raise ValueError(f"a must be [C={D.shape[0]}], got {tuple(a.shape)}")
    v = a.to(D.dtype) @ D                      # D^T a, [d]
    a2 = extract(v, D, lam1, lam2).a
    a64 = a.detach().to(torch.float64)
    a2_64 = a2.detach().to(torch.float64)
    rel_err = float(torch.linalg.norm(a2_64 - a64) /
                    (torch.linalg.norm(a64) + 1e-12))
    s1 = a64 > ACTIVE_TOL
    s2 = a2_64 > ACTIVE_TOL
    union = int((s1 | s2).sum())
    jac = 1.0 if union == 0 else float((s1 & s2).sum()) / union
    return ReextractResult(rel_err=rel_err, support_jaccard=jac, a2=a2)
