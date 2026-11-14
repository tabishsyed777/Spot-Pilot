"""Real subprocess checkpoint/resume demonstration; no cloud API calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def compute(state, step):
    for _ in range(2000):state=hashlib.sha256((state+':'+str(step)).encode()).hexdigest()
    return state


def atomic_write(path, data):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(data,f);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def run_worker(directory, steps=40, interval=5, fail_at=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'checkpoint.json'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(step=0,value='spotpilot')
    for step in range(state['step']+1,steps+1):
        state=dict(step=step,value=compute(state['value'],step))
        if fail_at==step:os._exit(23)
        if step%interval==0 or step==steps:atomic_write(checkpoint,state)
    return state


def demo(directory):
    path=Path(directory);path.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();script=Path(__file__).resolve()
    first=subprocess.run([sys.executable,str(script),'run','--directory',str(path),'--fail-at','17'],check=False)
    if first.returncode!=23:raise RuntimeError('expected injected worker exit 23')
    saved=json.loads((path/'checkpoint.json').read_text())
    second=subprocess.run([sys.executable,str(script),'run','--directory',str(path)],check=False,capture_output=True,text=True)
    if second.returncode:raise RuntimeError(second.stderr)
    recovered=json.loads((path/'checkpoint.json').read_text());expected='spotpilot'
    for step in range(1,41):expected=compute(expected,step)
    success=recovered['step']==40 and recovered['value']==expected
    result=dict(success=success,interrupted_at_step=17,resumed_from_step=saved['step'],recomputed_steps=17-saved['step'],total_steps=40,elapsed_seconds=time.perf_counter()-started,final_digest=recovered['value'],note='Real local process experiment; timing does not calibrate the scheduling simulator.')
    atomic_write(path/'result.json',result)
    if not success:raise RuntimeError('recovered workload differs from uninterrupted reference')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','demo']);p.add_argument('--directory',required=True);p.add_argument('--fail-at',type=int)
    a=p.parse_args();print(json.dumps(demo(a.directory) if a.command=='demo' else run_worker(a.directory,fail_at=a.fail_at),indent=2))

if __name__=='__main__':main()
