"""Smoke and fit receiver-local word tables on one explicitly granted node."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.manifest.read_text());root=Path(m['root']);manifest_sha=sha(a.manifest)
    assert socket.gethostname()==m['expected_host']
    for path,value in m['pins'].items():assert sha(Path(path))==value,path
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    (root/'logs').mkdir(exist_ok=True)
    state=dict(status='RUNNING',host=socket.gethostname(),controller_pid=os.getpid(),stages={})
    def save():
        state['utc']=datetime.now(timezone.utc).isoformat();temporary=root/'campaign_state.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,root/'campaign_state.json')
    def complete(phase,pair,shard):
        path=root/f'{phase}_p{pair}_s{shard}.done.json'
        if not path.exists():return False
        record=json.loads(path.read_text())
        assert record['status']=='COMPLETE_RECEIVER_LOCAL_FIT_ONLY' and record['manifest_sha256']==manifest_sha
        expected=len(m['engineering_words']) if phase=='engineering' else len(range(shard,221,4))
        assert record['words']==len(record['rows'])==expected
        for row in record['rows']:assert sha(Path(row['path']))==row['sha256']
        return True
    for phase in ['engineering','fit']:
        state['phase']=phase;state['stages'][phase]={};active={};save()
        for pair in [0,1]:
            for shard in range(1 if phase=='engineering' else 4):
                gpu=pair if phase=='engineering' else pair*4+shard;name=f'{phase}_p{pair}_s{shard}'
                if complete(phase,pair,shard):
                    state['stages'][phase][name]=dict(status='COMPLETE',gpu=gpu);save();continue
                log=(root/'logs'/f'{name}.log').open('a')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_PROGRESS_BARS='1')
                cmd=[sys.executable,m['runner'],'--manifest',str(a.manifest),'--phase',phase,'--pair',str(pair),'--shard',str(shard),'--shards','4']
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[name]=(proc,log,pair,shard);state['stages'][phase][name]=dict(status='RUNNING',gpu=gpu,pid=proc.pid);save()
                print('START',name,proc.pid,flush=True)
        while active:
            for name,(proc,log,pair,shard) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del active[name];good=code==0 and complete(phase,pair,shard)
                state['stages'][phase][name].update(status='COMPLETE' if good else 'FAILED',exit_code=code);save()
                print('END',name,state['stages'][phase][name]['status'],flush=True)
            if active:time.sleep(15)
        if any(x['status']!='COMPLETE' for x in state['stages'][phase].values()):
            state['status']='STOPPED_ON_FAILURE';save();return 1
    state['status']='COMPLETE_FIT_AWAITING_INDEPENDENT_AUDIT';save()
    print(state['status'],flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
