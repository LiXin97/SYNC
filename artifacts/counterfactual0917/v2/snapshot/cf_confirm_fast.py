"""Run the frozen confirmation with proven no-write forward reuse at gamma=0.

The policy explicitly bypasses all hooks at zero strength. Transmitted bytes
therefore cannot affect the forward. Preserve their metadata while reusing
the same public-prompt/no-hook computation; this is not a byte-budget or
latency claim. All nonzero interventions use the frozen runner unchanged.
"""
from pathlib import Path
import sys
R=next(p for p in Path(__file__).resolve().parents if (p/'README.md').exists())
sys.path.insert(0,str(R/'artifacts/counterfactual0917/v2/snapshot'))
import cf_confirm as frozen
original=frozen.Receiver.evaluate


def evaluate(self,scene,qi,packet=None,gamma=0,writer='band',extra=None,zero=False,replay=False):
    if packet is not None and gamma==0 and not zero:
        result=original(self,scene,qi,None,0,'band',extra,zero=False,replay=replay)
        raw=frozen.base.serialize(packet)
        result['payload_sha256']=frozen.base.sha(raw)
        result['payload_bytes']=len(raw)
        return result
    return original(self,scene,qi,packet,gamma,writer,extra,zero=zero,replay=replay)


frozen.Receiver.evaluate=evaluate
if __name__=='__main__':frozen.main()
