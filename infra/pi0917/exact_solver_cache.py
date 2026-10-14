"""Memoize identical solver inputs; no approximate reuse or arithmetic change.

This accelerates repeated controls, not a deployment latency claim. Periodic
hits are recomputed bit for bit. Keys cover the whole batch, its shape/dtype,
the dictionary tensor identity/version, and both penalties.
"""
from collections import OrderedDict
import hashlib
import torch


class ExactSolverCache:
    def __init__(self, solve, capacity=20000, verify_every=64):
        self.solve=solve;self.capacity=capacity;self.verify_every=verify_every
        self.cache=OrderedDict();self.hits=0;self.misses=0;self.verified=0

    def __call__(self,H,D,lam1,lam2):
        raw=H.detach().cpu().contiguous().numpy().tobytes()
        key=(tuple(H.shape),str(H.dtype),D.data_ptr(),D._version,tuple(D.shape),
             str(D.dtype),str(D.device),float(lam1),float(lam2),hashlib.sha256(raw).digest())
        if key in self.cache:
            self.hits+=1;entry=self.cache.pop(key);self.cache[key]=entry
            # Retain D itself: a reused allocator address can never collide.
            assert entry[0] is D
            value=entry[1].to(D.device)
            if self.hits % self.verify_every==0:
                check=self.solve(H,D,lam1,lam2)
                assert torch.equal(value,check),'exact solver cache replay failed'
                self.verified+=1
            return value
        self.misses+=1;value=self.solve(H,D,lam1,lam2)
        self.cache[key]=(D,value.detach().cpu().clone())
        if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return value
