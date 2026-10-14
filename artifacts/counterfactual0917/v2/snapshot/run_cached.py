from pathlib import Path
import sys,json
R=next(p for p in Path(__file__).resolve().parents if (p/'README.md').exists())
sys.path.insert(0,str(R/'infra/pi0917'))
from exact_solver_cache import ExactSolverCache
sys.path.insert(0,str(R/'artifacts/counterfactual0917/v2/snapshot'))
import counterfactual_run as base
import syncomm.writer as writer
cache=ExactSolverCache(writer._solve_a0,verify_every=64)
writer._solve_a0=cache
base.main()
print('CACHE_STATS',json.dumps(dict(hits=cache.hits,misses=cache.misses,verified=cache.verified)),flush=True)
