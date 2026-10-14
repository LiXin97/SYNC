"""Frozen engineering -> eight-worker discovery; no allocation or claim promotion."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest', type=Path, required=True); a = p.parse_args()
    m = json.loads(a.manifest.read_text()); root = Path(m['root'])
    assert socket.gethostname() == m['expected_host'], 'Node identity changed; inspect before resuming'
    for path, digest in m['pins'].items():
        assert sha(Path(path)) == digest, path
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not occupied, 'GPU processes exist; inspect ownership before launching'
    (root / 'logs').mkdir(exist_ok=True)
    state = dict(status='RUNNING', host=socket.gethostname(), controller_pid=os.getpid(), stages={})

    def save():
        state['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = root / 'campaign_state.tmp'; tmp.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(tmp, root / 'campaign_state.json')

    def complete(phase, pair, shard, expected):
        raw = root / 'raw' / f'{phase}_p{pair}_s{shard}.jsonl'
        marker = raw.with_suffix('.done.json')
        if not marker.exists():
            return False
        record = json.loads(marker.read_text())
        assert record['status'] == 'PASS' and record['rows'] == expected
        assert record['manifest_sha256'] == sha(a.manifest)
        assert record['raw_sha256'] == sha(raw)
        assert sum(1 for _ in raw.open()) == expected
        return True

    for phase in ('engineering', 'development'):
        shards = 1 if phase == 'engineering' else 4
        expected = 144 if phase == 'engineering' else 1440
        jobs = [(pair, shard, pair if phase == 'engineering' else pair * 4 + shard)
                for pair in (0, 1) for shard in range(shards)]
        active = {}
        state['phase'] = phase; state['stages'][phase] = {}; save()
        for pair, shard, gpu in jobs:
            name = f'{phase}_p{pair}_s{shard}'
            if complete(phase, pair, shard, expected):
                state['stages'][phase][name] = dict(status='COMPLETE', gpu=gpu); save(); continue
            log = (root / 'logs' / f'{name}.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_PROGRESS_BARS='1')
            cmd = [sys.executable, m['runner'], '--manifest', str(a.manifest), '--phase', phase,
                   '--pair', str(pair), '--shard', str(shard), '--shards', str(shards)]
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[name] = (proc, log, pair, shard)
            state['stages'][phase][name] = dict(status='RUNNING', gpu=gpu, pid=proc.pid); save()
            print('START', name, 'gpu', gpu, 'pid', proc.pid, flush=True)
        while active:
            for name, (proc, log, pair, shard) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close(); del active[name]
                good = code == 0 and complete(phase, pair, shard, expected)
                state['stages'][phase][name].update(status='COMPLETE' if good else 'FAILED', exit_code=code)
                save(); print('END', name, state['stages'][phase][name]['status'], flush=True)
            if active:
                time.sleep(15)
        if any(r['status'] != 'COMPLETE' for r in state['stages'][phase].values()):
            state['status'] = 'STOPPED_ON_FAILURE'; save(); return 1
    state['phase'] = 'analysis'; save()
    with (root / 'logs' / 'analysis.log').open('a') as log:
        result = subprocess.run([sys.executable, m['analyzer'], '--manifest', str(a.manifest)], stdout=log, stderr=subprocess.STDOUT)
    state['status'] = 'COMPLETE_DEVELOPMENT' if result.returncode == 0 and (root / 'development_analysis.json').exists() else 'ANALYSIS_FAILED'
    save(); print(state['status'], flush=True)
    return 0 if state['status'] == 'COMPLETE_DEVELOPMENT' else 1


if __name__ == '__main__':
    raise SystemExit(main())
