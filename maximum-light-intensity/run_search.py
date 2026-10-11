#!/usr/bin/env python3
"""Fault-tolerant parallel, adaptive multi-start four-hour illumination search."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, math, multiprocessing as mp, pathlib, queue, signal, sys, time, traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step")
from solver import oracle, grid_seed, intensity, optimize_exchange
from experiments import optimize_tracked, boundary_grid_seed, fast_oracle

ROOT=pathlib.Path(__file__).resolve().parent

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    return x

def atomic_json(path,data):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    with open(temp,'w') as f:
        json.dump(plain(data),f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def worker_loop(wid,inbox,outbox):
    while True:
        task=inbox.get()
        if task is None:return
        start=time.time()
        try:
            p=np.asarray(task['points'])
            if task['method']=='exchange':
                result=optimize_exchange(p,rounds=90)
            else:
                result=optimize_tracked(p,maxiter=task.get('maxiter',180),rounds=6,
                         radius=task.get('radius',.045),grid=41,ftol=1e-10,
                         oracle_fn=fast_oracle if task['id']%10 else None)
            p=np.asarray(result['points'])
            if len(p)!=task['n'] or not np.all(np.isfinite(p)) or np.min(p)<0 or np.max(p)>1:
                raise ValueError('Invalid light coordinates')
            vals,sites=oracle(p,grid=61,extra=result['witnesses'])
            # Independently check prospective improvements before checkpoint promotion.
            # This feeds missed minima back into selection instead of merely logging them.
            if vals[0] > task.get('incumbent_objective', 0.) + 1e-7:
                from validation import validate
                cert = validate(p, tolerance=1e-5, time_limit=60.,
                                numerical=False, witness=sites[0])
                cert['coordinates_sha256_float64_little_endian'] = hashlib.sha256(np.asarray(p, dtype='<f8').tobytes()).hexdigest()
                cert['validator_source_sha256'] = hashlib.sha256((ROOT/'validation.py').read_bytes()).hexdigest()
                cert['claimed_objective_before_validation'] = float(vals[0])
                vals, sites = oracle(p, grid=61, extra=np.vstack((sites, cert['witness'])))
                result['validation'] = cert
            result.update(n=len(p),objective=float(vals[0]),witnesses=sites,
                          witness_values=vals,metadata={'task':task['id'],'seed_kind':task['kind'],
                          'rng_seed':task['rng_seed'],'method':task['method'],
                          'elapsed':time.time()-start,'history':result.get('history',result.get('metadata',{}).get('history',[])),
                          'completed_at':time.time(),'constraint_check':{'finite':True,'count':len(p),'inside_unit_square':True},
                          'objective_status':'independently bounded' if result.get('validation', {}).get('certified') else 'numerical; continuous domain bound is in validation.json'})
            outbox.put({'worker':wid,'task':task['id'],'result':plain(result),'elapsed':time.time()-start})
        except BaseException:
            outbox.put({'worker':wid,'task':task['id'],'error':traceback.format_exc(),'elapsed':time.time()-start})


def read_best(checkpoints):
    best={}
    for n in range(36,51):
        path=checkpoints/f'n{n}'/'best.json'
        if path.exists():
            try:best[n]=json.loads(path.read_text())
            except (ValueError,OSError):pass
    return best


def generate_seed(n,best,pools,rng,count):
    seed36=json.loads((ROOT/'source/seed36.json').read_text())['points']
    modes=['jitter','jitter','shake','move','neighbor','neighbor','ring','grid','pool','reference']
    kind=modes[int(rng.integers(len(modes)))]
    if n not in best:kind='reference' if count%3 else 'ring'
    if kind=='reference':
        p=np.asarray(seed36).copy()
        while len(p)<n:
            # Insert into an under-lit interior cell, with randomized tie breaking.
            test=rng.uniform(.1,.9,(600,2));v=intensity(p,test)
            choices=np.argsort(v)[:max(2,20)]
            p=np.vstack((p,test[rng.choice(choices)]))
        p+=rng.normal(0,.002,p.shape)
    elif kind=='ring':
        p=boundary_grid_seed(n,rng,boundary_fraction=rng.uniform(.48,.70))
    elif kind=='grid':
        p=grid_seed(n,rng,int(rng.integers(9)))
        # Expand some seeds toward the edges to reduce corner deficit.
        if rng.random()<.75:p=.5+(p-.5)*rng.uniform(1.08,1.16)
    elif kind=='neighbor' and any(k in best for k in (n-1,n+1)):
        k=int(rng.choice([k for k in (n-1,n+1) if k in best]))
        p=np.array(best[k]['points'])
        if k<n:
            ws=np.array(best[k]['witnesses'])
            if rng.random()<.35:
                ws=ws[np.any((ws<.01)|(ws>.99),axis=1)]
                ws=np.clip(ws,.025,.975)
            else:
                ws=ws[np.all((ws>.05)&(ws<.95),axis=1)]
            if not len(ws):ws=rng.uniform(.1,.9,(100,2))
            p=np.vstack((p,ws[rng.integers(len(ws))]+rng.normal(0,.012,2)))
        else:
            ws=np.array(best[k]['witnesses']);v=intensity(p,ws)
            d2=np.sum((ws[:,None,:]-p[None,:,:])**2,axis=2)
            removal=np.min(v[:,None]-1/np.maximum(d2,1e-20),axis=0)
            choices=np.argsort(removal)[-min(5,len(p)):]
            p=np.delete(p,rng.choice(choices),axis=0)
        p+=rng.normal(0,.002,p.shape)
    else:
        if n not in best:return generate_seed(n,best,pools,rng,1)
        candidate=best[n]
        if kind=='pool' and pools.get(n):candidate=pools[n][rng.integers(len(pools[n]))]
        p=np.array(candidate['points'])
        if kind in ('jitter','pool'):p+=rng.normal(0,rng.choice([.0005,.002,.006,.012]),p.shape)
        elif kind=='shake':p+=rng.normal(0,rng.choice([.02,.04,.07]),p.shape)
        elif kind=='move':
            j=rng.choice(n,size=int(rng.integers(1,4)),replace=False)
            if rng.random()<.5:p[j]+=rng.normal(0,.12,(len(j),2))
            else:
                ws=np.array(candidate['witnesses'])
                p[j]=ws[rng.choice(len(ws),size=len(j))]+rng.normal(0,.035,(len(j),2))
        elif kind=='neighbor':p+=rng.normal(0,.02,p.shape)
    return np.clip(p,.00001,.99999),kind


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=4);ap.add_argument('--workers',type=int,default=20)
    ap.add_argument('--run-dir',type=pathlib.Path,default=ROOT/'runs/four-hour');ap.add_argument('--seed',type=int,default=20261002)
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True);checkpoints=run/'checkpoints'
    manifest_path=run/'run.json'
    start=time.time()
    if args.resume and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());deadline=manifest['deadline'];start=manifest['started_at']
    else:
        deadline=start+args.hours*3600
        manifest={'started_at':start,'deadline':deadline,'requested_seconds':args.hours*3600,'workers':args.workers,
                  'seed':args.seed,'status':'running','problem':'max min inverse square intensity on [0,1]^2',
                  'python':sys.version,'sources_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}}
        atomic_json(manifest_path,manifest)
    rng=np.random.default_rng(args.seed+int(time.time()-start))
    best=read_best(checkpoints);pools={n:([b] if (b:=best.get(n)) else []) for n in range(36,51)}
    attempts={n:0 for n in range(36,51)};seconds={n:0. for n in range(36,51)};gains={n:0.002 for n in range(36,51)}
    ctx=mp.get_context('spawn');outbox=ctx.Queue();workers={};active={};taskid=0;completed=0;failures=0;last_status=0.
    def spawn(wid):
        inbox=ctx.Queue();proc=ctx.Process(target=worker_loop,args=(wid,inbox,outbox),daemon=True);proc.start()
        workers[wid]=(proc,inbox)
    for wid in range(args.workers):spawn(wid)
    events=open(run/'events.jsonl','a',buffering=1)
    def record(event):events.write(json.dumps(plain({'at':time.time(),**event}),allow_nan=False)+'\n')
    record({'event':'start' if not args.resume else 'resume','deadline':deadline,'workers':args.workers})
    try:
        while time.time()<deadline:
            for wid,(proc,inbox) in list(workers.items()):
                if not proc.is_alive() or (wid in active and time.time()-active[wid]['started']>240):
                    old=active.pop(wid,None)
                    if proc.is_alive():proc.terminate()
                    proc.join(timeout=1);record({'event':'worker_restart','worker':wid,'task':old});failures+=1;spawn(wid)
                if wid not in active and time.time()<deadline-5:
                    ongoing={n:sum(t['n']==n for t in active.values()) for n in range(36,51)}
                    coverage=[n for n in range(36,51) if attempts[n]+ongoing[n]<2]
                    if coverage:n=min(coverage,key=lambda n:attempts[n]+ongoing[n])
                    elif rng.random()<.45:n=min(range(36,51),key=lambda n:seconds[n]+ongoing[n]*20)
                    else:
                        weights=np.array([.35+min(3,200*gains[n]) for n in range(36,51)])
                        weights[0]*=1.3
                        weights/=np.array([1+.35*ongoing[n] for n in range(36,51)])
                        n=int(rng.choice(np.arange(36,51),p=weights/weights.sum()))
                    rs=int(rng.integers(2**32));p,kind=generate_seed(n,best,pools,np.random.default_rng(rs),taskid)
                    method='tracked' if rng.random()<.94 else 'exchange'
                    task={'id':taskid,'n':n,'points':p.tolist(),'kind':kind,'method':method,'rng_seed':rs,
                          'incumbent_objective':best.get(n, {}).get('objective', 0.),
                          'maxiter':220 if kind in ('jitter','pool') else 180,'radius':.04 if kind in ('jitter','pool') else .065}
                    workers[wid][1].put(task);active[wid]={'id':taskid,'n':n,'started':time.time(),'kind':kind};taskid+=1
            try:message=outbox.get(timeout=.5)
            except queue.Empty:message=None
            if message:
                wid=message['worker'];task=active.get(wid)
                if task and task['id']==message['task']:
                    active.pop(wid);n=task['n'];attempts[n]+=1;seconds[n]+=message['elapsed'];completed+=1
                    if 'error' in message:failures+=1;record({'event':'error',**message})
                    else:
                        b=message['result'];old=best.get(n);gain=max(0,b['objective']-(old['objective'] if old else 0))
                        gains[n]=.97*gains[n]+.03*gain/max(b['objective'],1)
                        record({'event':'candidate','task':task['id'],'n':n,'objective':b['objective'],'elapsed':message['elapsed'],'kind':task['kind'],'improved':gain>1e-8})
                        if old is None or b['objective']>old['objective']+1e-7:
                            atomic_json(checkpoints/f'n{n}'/'best.json',b);best[n]=b
                            if b.get('validation'):
                                atomic_json(checkpoints/f'n{n}'/'validation.json',b['validation'])
                            atomic_json(checkpoints/f'n{n}'/'history'/f'{task["id"]:07d}.json',b)
                            print(f'BEST n={n} I={b["objective"]:.10f} gain={gain:.6g} task={task["id"]} {task["kind"]}',flush=True)
                        pool=pools[n];pool.append(b);pool.sort(key=lambda x:x['objective'],reverse=True)
                        # Keep representatives separated in objective to retain alternate basins.
                        selected=[]
                        for item in pool:
                            if not selected or all(abs(item['objective']-x['objective'])>2e-4 for x in selected):selected.append(item)
                            if len(selected)>=8:break
                        pools[n]=selected
            if time.time()-last_status>30:
                status={'elapsed_seconds':time.time()-start,'remaining_seconds':max(0,deadline-time.time()),'completed':completed,'failures':failures,
                        'active':active,'attempts':attempts,'worker_seconds':seconds,'best':{n:b['objective'] for n,b in best.items()}}
                atomic_json(run/'status.json',status);last_status=time.time()
                print(f'STATUS elapsed={(time.time()-start)/60:.1f}min completed={completed} failures={failures} covered={len(best)}/15',flush=True)
    finally:
        for wid,(proc,inbox) in workers.items():
            if proc.is_alive():proc.terminate()
            proc.join(timeout=2)
        record({'event':'stop','elapsed_seconds':time.time()-start,'completed':completed,'failures':failures})
        events.close();manifest.update(status='completed' if time.time()>=deadline else 'interrupted',finished_at=time.time(),elapsed_seconds=time.time()-start,completed=completed,failures=failures,attempts=attempts,worker_seconds=seconds)
        atomic_json(manifest_path,manifest)
        atomic_json(run/'status.json',{'status':manifest['status'],'elapsed_seconds':manifest['elapsed_seconds'],
                    'remaining_seconds':max(0.,deadline-time.time()),'completed':completed,'failures':failures,
                    'active':{},'attempts':attempts,'worker_seconds':seconds,'best':{n:b['objective'] for n,b in best.items()}})
    print('SEARCH COMPLETE',flush=True)

if __name__=='__main__':main()
