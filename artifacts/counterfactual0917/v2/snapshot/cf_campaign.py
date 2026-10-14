"""Bounded eight-GPU confirmation queue with artifact-based completion checks."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,time
ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);a=ap.parse_args()
m=json.loads(a.manifest.read_text());root=Path(m['root']);runner=Path(m['runner']);shards=8
for p,h in m['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
bank=[s for s in json.loads(Path(m['bank']).read_text()) if s['split']=='eval'];assert len(bank)==720
jobs=[]
for shard in range(shards):
 for pair in range(4):
  selected=bank[shard::shards]
  match=json.loads((root/f"matching_eval_{m['pairs'][pair]['sender']}.json").read_text())['assignment']
  expected=sum(36+4*(s['scene'] in m['reference_scenes'])+8*(s['scene'] in match) for s in selected)
  jobs.append(dict(id=f'p{pair}_s{shard}',pair=pair,shard=shard,expected_rows=expected))
active={};finished=set();failed={};started=set();logs={}

def complete(job):
 raw=root/'confirmation_raw'/f'eval_p{job["pair"]}_s{job["shard"]}.jsonl'
 marker=raw.with_suffix('.done.json')
 if not marker.exists():return False
 r=json.loads(marker.read_text());assert r['status']=='PASS' and r['rows']==job['expected_rows'],job
 assert sum(1 for _ in raw.open())==r['rows'],job
 return True

for j in jobs:
 if complete(j):finished.add(j['id'])
(root/'logs').mkdir(exist_ok=True)
while len(finished)+len(failed)<len(jobs):
 for gpu,(proc,j) in list(active.items()):
  if proc.poll() is None:continue
  logs[j['id']].close();del active[gpu]
  if proc.returncode==0 and complete(j):finished.add(j['id']);print('COMPLETE',j['id'],flush=True)
  else:failed[j['id']]=dict(exit_code=proc.returncode,reason='Missing validated artifact or process failure');print('FAILED',j['id'],failed[j['id']],flush=True)
 for gpu in range(8):
  if gpu in active:continue
  todo=next((j for j in jobs if j['id'] not in finished|started|set(failed)),None)
  if todo is None:continue
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
  log=(root/'logs'/f'confirm_{todo["id"]}.log').open('a');logs[todo['id']]=log
  cmd=[sys.executable,str(runner),'--manifest',str(a.manifest),'--phase','eval','--pair',str(todo['pair']),'--shard',str(todo['shard']),'--shards',str(shards)]
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  active[gpu]=(proc,todo);started.add(todo['id']);print('START',todo['id'],'GPU',gpu,'PID',proc.pid,flush=True)
 state=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),expected_jobs=jobs,
            complete=sorted(finished),failed=failed,running={str(gpu):dict(job=j['id'],pid=p.pid) for gpu,(p,j) in active.items()})
 tmp=root/'campaign_state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,root/'campaign_state.json')
 if active:time.sleep(20)
 elif len(finished)+len(failed)<len(jobs):raise RuntimeError('Queue lost a job')
if failed:raise SystemExit('Incomplete confirmation; see campaign_state.json and retained logs')
cmd=[sys.executable,m['analyzer'],'--root',str(root),'--manifest',str(a.manifest),'--shards','8']
with (root/'logs/analysis.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
assert (root/'confirmation_analysis.json').exists()
print('ALL CONFIRMATION ROWS AND LOCKED ANALYSIS COMPLETE',flush=True)
