"""Continuous minimax search for unit-square inverse-square illumination."""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.ndimage import minimum_filter
from scipy.spatial import Voronoi, QhullError


def intensity(points, sites):
    p, q = np.asarray(points), np.atleast_2d(sites)
    d = q[:, None, :] - p[None, :, :]
    r2 = np.sum(d*d, axis=2)
    with np.errstate(divide='ignore'):
        return (1/r2).sum(axis=1)


def intensity_jac(points, sites):
    d = np.atleast_2d(sites)[:, None, :] - np.asarray(points)[None, :, :]
    r2 = np.maximum(np.sum(d*d, axis=2), 1e-24)
    return (1/r2).sum(axis=1), (2*d/r2[:, :, None]**2).reshape(len(d), -1)


def site_value_grad(q, points, scale=1.):
    d = q-points
    r2 = np.maximum(np.sum(d*d,axis=1), 1e-28)
    return np.sum(1/r2)/scale, np.sum(-2*d/r2[:,None]**2,axis=0)/scale


def unique_sites(sites, eps=1e-6):
    keep=[]
    for q in sites:
        if not keep or np.min(np.linalg.norm(np.asarray(keep)-q,axis=1))>eps:
            keep.append(q)
    return np.array(keep).reshape(-1,2)


def oracle(points, grid=39, extra=None):
    """Discover interior/edge minima and include every corner; numerical only."""
    p=np.asarray(points,dtype=float)
    t=np.linspace(0,1,grid)
    xx, yy=np.meshgrid(t,t)
    sites=np.column_stack((xx.ravel(),yy.ravel()))
    vals=intensity(p,sites).reshape(grid,grid)
    mask=(vals==minimum_filter(vals,size=3,mode='constant',cval=np.inf))
    mask[[0,-1],:]=False; mask[:,[0,-1]]=False
    starts=[q for q in sites[mask.ravel()]]
    try:
        vv=Voronoi(p).vertices
        starts.extend(vv[np.all((vv>0)&(vv<1),axis=1)])
    except QhullError:
        pass
    if extra is not None:
        starts.extend(q for q in np.asarray(extra) if np.all((q>1e-8)&(q<1-1e-8)))
    starts=unique_sites(starts,eps=.009)
    minima=[np.array([0.,0.]), np.array([0.,1.]),np.array([1.,0.]),np.array([1.,1.])]
    scale=len(p)*10.
    for q in starts:
        res=minimize(site_value_grad,q,args=(p,scale),jac=True,method='L-BFGS-B',
                     bounds=[(0,1)]*2,options={'ftol':1e-14,'gtol':1e-9,'maxiter':80,'maxls':25})
        minima.append(res.x)
    # Every edge is searched independently, so no projection can omit a boundary minimum.
    edge_t=np.linspace(0,1,max(grid*3,121))
    for axis in (0,1):
        for fixed in (0.,1.):
            qs=np.zeros((len(edge_t),2)); qs[:,axis]=edge_t; qs[:,1-axis]=fixed
            vs=intensity(p,qs)
            indices=np.where((vs[1:-1]<=vs[:-2])&(vs[1:-1]<=vs[2:]))[0]+1
            for j in indices:
                def fun(t):
                    z=np.zeros(2);z[axis]=t;z[1-axis]=fixed
                    return intensity(p,[z])[0]
                res=minimize_scalar(fun,bounds=(edge_t[j-1],edge_t[j+1]),method='bounded',options={'xatol':1e-14})
                z=np.zeros(2);z[axis]=res.x;z[1-axis]=fixed
                minima.append(z)
    minima=unique_sites(minima,eps=1e-5)
    values=intensity(p,minima)
    order=np.argsort(values)
    return values[order],minima[order]


def optimize_exchange(points, rounds=25, maxiter=120, ftol=2e-9, deadline=None):
    p=np.asarray(points).copy(); n=len(p); scale=n*10.
    vals,sites=oracle(p)
    best=(vals[0],p.copy(),sites.copy())
    history=[]
    radius=.025
    for iteration in range(rounds):
        if deadline and time.time()>deadline: break
        v0=vals[0]
        x0=np.r_[p.ravel(),v0/scale]
        cache={}
        def quantities(x):
            if 'x' not in cache or not np.array_equal(cache['x'],x):
                v,j=intensity_jac(x[:-1].reshape(n,2),sites)
                cache.update(x=x.copy(),v=v/scale-x[-1],j=np.column_stack((j/scale,-np.ones(len(v)))))
            return cache
        res=minimize(lambda x:-x[-1],x0,jac=lambda x:np.r_[np.zeros(2*n),-1.],
                     bounds=[(max(0.,v-radius),min(1.,v+radius)) for v in p.ravel()]+[(0,None)],method='SLSQP',
                     constraints={'type':'ineq','fun':lambda x:quantities(x)['v'],'jac':lambda x:quantities(x)['j']},
                     options={'ftol':ftol,'maxiter':maxiter})
        newp=res.x[:-1].reshape(n,2)
        nv,ns=oracle(newp,extra=sites)
        history.append([int(iteration),float(nv[0]),float(res.x[-1]*scale),int(res.nit),bool(res.success)])
        if nv[0]>best[0]:best=(float(nv[0]),newp.copy(),ns.copy())
        ratio=(nv[0]-v0)/max(1e-8,res.x[-1]*scale-v0)
        if nv[0]>=v0-1e-8:
            p=newp;vals=nv
            if ratio>.75:radius=min(.035,radius*1.5)
            elif ratio<.25:radius=max(2e-6,radius*.5)
        else:
            radius=max(2e-6,radius*.45)
        # Retain only most recent two contact sets; old sites consume constraints but stabilize exchange.
        sites=unique_sites(np.vstack((ns,sites)),eps=5e-5)
        sv=intensity(p,sites)
        sites=sites[sv<max(nv[0]*1.25,res.x[-1]*scale*1.1)]
        if len(sites)>6*n:sites=sites[np.argsort(intensity(p,sites))[:6*n]]
        if radius<5e-6 and abs(nv[0]-v0)<2e-6:break
    value,p,sites=best
    vals,sites=oracle(p,grid=49,extra=sites)
    return {'n':n,'points':p.tolist(),'objective':float(vals[0]),'witnesses':sites.tolist(),
            'metadata':{'method':'exchange-slsqp','history':history}}


def grid_seed(n, rng, variant=0):
    """Balanced rows, optionally staggered; no dependence on published coordinates."""
    rows=max(2,round(np.sqrt(n))+int(variant%3)-1)
    sizes=np.full(rows,n//rows);sizes[:n%rows]+=1
    if variant%2:sizes=sizes[rng.permutation(rows)]
    ps=[]
    for i,k in enumerate(sizes):
        y=(i+.5)/rows
        xs=(np.arange(k)+.5)/k
        if variant%4==2: xs+=(.13/k)*(-1 if i%2 else 1)
        ps.extend((x,y) for x in xs)
    p=np.array(ps)
    if variant>=4:p+=rng.normal(0,.012,p.shape)
    return np.clip(p,.012,.988)

if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=36);ap.add_argument('--rounds',type=int,default=12)
    a=ap.parse_args();t=time.time();res=optimize_exchange(grid_seed(a.n,np.random.default_rng(42)),rounds=a.rounds)
    print(json.dumps({'objective':res['objective'],'elapsed':time.time()-t,'history':res['metadata']['history']}))
