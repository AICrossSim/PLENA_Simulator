#!/usr/bin/env python3
"""Run the complete numerical matrix after both required real captures finish.

This is an explicit durable job launcher, not a fabricated completion marker.
It saves the invoked command and exit state; evaluation itself rejects incomplete
or overlapping captures and freezes only configurations that meet accuracy.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--capture-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--expected-tensors',type=Path)
    ap.add_argument('--rank-lanes',default='8,16',help='shared resident bases for8,16')
    ap.add_argument('--resume-q1',type=Path)
    ap.add_argument('--resume-q3',type=Path)
    ap.add_argument('--timeout-hours',type=float,default=8);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();cal=args.capture_root/'development/manifest.json';ev=args.capture_root/'heldout/manifest.json'
    state=args.output/'job_status.json'
    lock=(args.output/'launcher.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('another numerical launcher owns this output directory')
    if state.exists():
        previous=json.loads(state.read_text())
        pid=previous.get('pid')
        if previous.get('status')=='running_full_Q0_Q3' and pid:
            try:os.kill(int(pid),0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'active numerical evaluator PID {pid} already owns this output directory')
    while True:
        counts=[];ready=True
        for p,needed in ((cal,32768),(ev,8192)):
            try:
                m=json.loads(p.read_text());n=m.get('token_count',0);ready &= bool(m.get('complete')) and n>=needed
            except (FileNotFoundError,json.JSONDecodeError):n=0;ready=False
            counts.append(n)
        state.write_text(json.dumps({'status':'waiting_for_real_capture','token_counts':counts,'needed':[32768,8192],
                                    'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
        if ready:break
        if time.monotonic()-start>args.timeout_hours*3600:raise TimeoutError('capture did not complete; Q0 remains incomplete')
        time.sleep(15)
    command=[sys.executable,str(Path(__file__).with_name('evaluate.py')),'--model',args.model,'--calibration',str(cal),
             '--evaluation',str(ev),'--output',str(args.output),'--rank-lanes',str(args.rank_lanes)]
    if args.expected_tensors:command += ['--expected-tensors',str(args.expected_tensors)]
    if args.resume_q1:command += ['--resume-q1',str(args.resume_q1)]
    if args.resume_q3:command += ['--resume-q3',str(args.resume_q3)]
    state.write_text(json.dumps({'status':'running_full_Q0_Q3','command':command,'token_counts':counts},indent=2)+'\n')
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with (args.output/'evaluate.log').open('w') as log:
        r=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        state.write_text(json.dumps({'status':'running_full_Q0_Q3','command':command,'pid':r.pid,
                                    'token_counts':counts},indent=2)+'\n')
        r.wait()
    state.write_text(json.dumps({'status':'completed' if r.returncode==0 else 'failed','exit_code':r.returncode,
                                'command':command,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
    sys.exit(r.returncode)


if __name__=='__main__':main()
