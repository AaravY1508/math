#!/usr/bin/env python3
"""Rerank archived search candidates and repair missed continuous minima.

This is a bounded post-search correctness repair, separate from the completed
four-hour search. Immutable search histories are never modified. Candidate
selection uses the repaired numerical oracle; final selection uses independent
outward-rounded whole-square certificates, with certificate witnesses fed back
into local optimization whenever the numerical oracle overstates performance.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent


def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(plain(data),indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def coordinate_hash(points):
    return hashlib.sha256(np.asarray(points,dtype='<f8').tobytes()).hexdigest()


def inputs_for_n(n):
    paths=[]
    for run in ('four-hour','benchmark'):
        base=HERE/'runs'/run/'checkpoints'/f'n{n}'
        paths.extend(base.glob('history/*.json'))
        paths.extend(base.glob('validations/*/checkpoint.json'))
        if (base/'best.json').exists():paths.append(base/'best.json')
    candidates={}
    for path in sorted(paths):
        data=json.loads(path.read_text())
        key=coordinate_hash(data['points'])
        if key not in candidates:
            candidates[key]={'data':data,'sources':[],'hash':key}
        candidates[key]['sources'].append(str(path.relative_to(HERE)))
    return list(candidates.values())


def finalize_n(n,out,deadline,polish_count=4,tolerance=1e-6):
    from solver import oracle,unique_sites,intensity
    from experiments import optimize_tracked
    from validation import validate,numerical_minimum
    start=time.time();out=Path(out)/f'n{n}';out.mkdir(parents=True,exist_ok=True)
    records=[];known_witnesses=[]
    for run in ('four-hour','benchmark'):
        base=HERE/'runs'/run/'checkpoints'/f'n{n}'
        for path in list(base.glob('validation.json'))+list(base.glob('validations/*/validation.json')):
            data=json.loads(path.read_text())
            if data.get('witness') is not None:known_witnesses.append(data['witness'])
    def robust_oracle(p,grid=61,extra=None):
        all_extra=list(known_witnesses)
        if extra is not None:all_extra.extend(np.asarray(extra).tolist())
        return oracle(p,grid=max(61,grid),extra=all_extra or None)
    ranked=[]
    for candidate in inputs_for_n(n):
        data=candidate['data'];p=np.asarray(data['points'])
        vals,sites=robust_oracle(p,extra=data.get('witnesses'))
        current={'n':n,'points':p,'objective':float(vals[0]),'witnesses':sites,
                 'witness_values':vals,'metadata':{'method':'post-search-rerank',
                 'source_files':candidate['sources'],'original_claimed_objective':data.get('objective'),
                 'original_coordinate_hash':candidate['hash']}}
        ranked.append(current)
        records.append({'stage':'rerank','sources':candidate['sources'],'hash':candidate['hash'],
                        'claimed':data.get('objective'),'corrected':float(vals[0]),
                        'drop':float(data.get('objective',vals[0])-vals[0])})
        write_json(out/'candidates'/f"original-{candidate['hash']}.json",current)
    ranked.sort(key=lambda c:c['objective'],reverse=True)
    # Retain the strongest distinct numerical basins and nominal leaders: the
    # latter can recover after their omitted minimum is made an active constraint.
    selected=[]
    for c in ranked:
        if not selected or all(abs(c['objective']-s['objective'])>1e-5 for s in selected):selected.append(c)
        if len(selected)>=polish_count:break
    nominal=sorted(ranked,key=lambda c:c['metadata']['original_claimed_objective'],reverse=True)
    for c in nominal[:2]:
        if all(coordinate_hash(c['points'])!=coordinate_hash(s['points']) for s in selected):selected.append(c)
    contenders=ranked[:2]
    for ci,c in enumerate(selected):
        if time.time()>deadline-90:break
        r=optimize_tracked(c['points'],maxiter=350,rounds=8,radius=.025,grid=61,
                           ftol=2e-12,oracle_fn=robust_oracle)
        vals,sites=robust_oracle(r['points'],grid=81,extra=r['witnesses'])
        polished={'n':n,'points':r['points'],'objective':float(vals[0]),'witnesses':sites,
                  'witness_values':vals,'metadata':{'method':'post-search-robust-polish',
                  'parent':c['metadata'],'optimizer_history':r['history'],
                  'polish_elapsed':r['elapsed']}}
        contenders.append(polished)
        records.append({'stage':'polish','candidate':ci,'before':c['objective'],
                        'after':polished['objective'],'elapsed':r['elapsed']})
        write_json(out/'candidates'/f'polished-{ci}.json',polished)
    contenders.sort(key=lambda c:c['objective'],reverse=True)
    # Certify competitors only while their numerical value can beat the current
    # best certified lower bound; each independently discovered witness is kept.
    best=None;best_cert=None;certifications=[]
    for ci,c in enumerate(contenders):
        if best_cert is not None and c['objective']<best_cert['objective_lower']-1e-7:continue
        p=np.asarray(c['points'])
        for repair in range(4):
            if time.time()>deadline-10 and best is not None:break
            witness=np.asarray(c['witnesses'])[0]
            limit=max(5.,min(65.,deadline-time.time()-3))
            cert=validate(p,tolerance=tolerance,time_limit=limit,numerical=False,witness=witness)
            cert.update({'validator_source_sha256':hashlib.sha256((HERE/'validation.py').read_bytes()).hexdigest(),
                         'verified_at_utc':datetime.now(timezone.utc).isoformat(),
                         'claimed_objective':c['objective'],
                         'claimed_objective_minus_proved_upper':c['objective']-cert['objective_upper'],
                         'oracle_missed_lower_minimum':c['objective']-cert['objective_upper']>max(tolerance,1e-6)})
            records.append({'stage':'certificate','candidate':ci,'repair':repair,
                            'numerical':c['objective'],'lower':cert['objective_lower'],
                            'upper':cert['objective_upper'],'certified':cert['certified'],
                            'elapsed':cert['elapsed_seconds']})
            certifications.append({'candidate':ci,'repair':repair,'certificate':cert})
            known_witnesses.append(cert['witness'])
            if best_cert is None or cert['objective_lower']>best_cert['objective_lower']:
                best,best_cert=c.copy(),cert.copy()
                best['objective']=min(c['objective'],cert['objective_upper'])
                best['metadata']={**best['metadata'],'objective_status':'independently certified continuous minimum',
                                  'post_search_finalization':True}
                write_json(out/'best.json',best)
                write_json(out/'validation.json',best_cert)
            if not cert['oracle_missed_lower_minimum']:break
            # Proof-generated witnesses make the missing basin an explicit active
            # branch. A smaller movement radius preserves it during recovery.
            r=optimize_tracked(p,maxiter=400,rounds=10,radius=.018,grid=81,
                              ftol=2e-12,oracle_fn=robust_oracle)
            vals,sites=robust_oracle(r['points'],grid=101,extra=r['witnesses'])
            c={'n':n,'points':r['points'],'objective':float(vals[0]),'witnesses':sites,
               'witness_values':vals,'metadata':{'method':'certificate-feedback-repair',
               'parent':c['metadata'],'optimizer_history':r['history'],
               'proof_witness':cert['witness']}}
            p=np.asarray(c['points'])
            write_json(out/'candidates'/f'repair-{ci}-{repair}.json',c)
    report={'n':n,'input_count':len(ranked),'polished_count':sum(r['stage']=='polish' for r in records),
            'best_objective':best['objective'] if best else None,
            'certificate_lower':best_cert['objective_lower'] if best_cert else None,
            'certified':bool(best_cert and best_cert['certified']),
            'elapsed_seconds':time.time()-start,'decisions':records,'certifications':certifications}
    write_json(out/'decisions.json',report)
    return {k:v for k,v in report.items() if k not in ('decisions','certifications')}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=HERE/'results'/'final-checkpoints')
    ap.add_argument('--workers',type=int,default=10)
    ap.add_argument('--seconds',type=float,default=540.)
    ap.add_argument('--polish-count',type=int,default=3)
    ap.add_argument('--tolerance',type=float,default=1e-6)
    a=ap.parse_args();start=time.time();deadline=start+a.seconds
    a.output.mkdir(parents=True,exist_ok=True)
    manifest={'started_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'post-search correctness repair',
              'source_runs':['runs/four-hour','runs/benchmark'],'deadline_unix':deadline,
              'maximum_workers':a.workers,'tolerance':a.tolerance,'results':[]}
    write_json(a.output/'finalization.json',manifest)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        order=[39,40,43,47,50,36,37,38,41,42,44,45,46,48,49]
        futures={pool.submit(finalize_n,n,a.output,deadline,a.polish_count,a.tolerance):n for n in order}
        for future in as_completed(futures):
            try:result=future.result()
            except BaseException:result={'n':futures[future],'error':traceback.format_exc()}
            manifest['results'].append(result)
            print(json.dumps(result),flush=True)
            write_json(a.output/'finalization.json',manifest)
    manifest.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
    write_json(a.output/'finalization.json',manifest)

if __name__=='__main__':main()
