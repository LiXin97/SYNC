"""Maximum-coverage role-balanced matching using only admissible edges.

Self edges represent excluded examples, never delivered placebo messages.
Minimizing their count maximizes coverage; bounded quality costs break ties.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def strict_derangement(keys, feasible, cost=None):
    keys = list(keys)
    n = len(keys)
    if not n:
        return {}, []
    index = {key:i for i,key in enumerate(keys)}
    assert len(index) == n
    matrix = np.full((n,n), float(n+1))
    np.fill_diagonal(matrix, 1.)
    for r in keys:
        for d in feasible.get(r, ()):
            if d == r or d not in index:
                continue
            c = max(0., float(cost(r,d))) if cost else 0.
            assert np.isfinite(c)
            matrix[index[r],index[d]] = (c/(1+c))/(n+1)
    rows, cols = linear_sum_assignment(matrix)
    assignment = {keys[i]:keys[j] for i,j in zip(rows,cols) if i != j}
    assert set(assignment) == set(assignment.values())
    assert all(d in feasible[r] for r,d in assignment.items())
    return assignment, [key for key in keys if key not in assignment]
