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


def _site_newton(q, points, maxiter=70):
    """Trust-limited damped Newton descent that stays in a seed's local basin.

    An unrestricted first quasi-Newton step can jump from a low grid sample
    across a narrow minimum into a shallower basin. Exact two-dimensional
    curvature and a radius tied to the closest pole avoid that failure.
    """
    q = np.clip(np.asarray(q, dtype=float), 0., 1.).copy()
    for _ in range(maxiter):
        delta = q-points
        r2 = np.maximum(np.sum(delta*delta, axis=1), 1e-28)
        inv = 1/r2
        value = float(inv.sum())
        gradient = -2*np.sum(delta*inv[:, None]**2, axis=0)
        hessian = (8*np.einsum('ni,nj,n->ij', delta, delta, inv**3)
                   - 2*np.eye(2)*np.sum(inv**2))
        free = ~(((q <= 0) & (gradient >= 0)) | ((q >= 1) & (gradient <= 0)))
        if not np.any(free) or np.max(np.abs(gradient[free])) < 1e-7:
            break
        h = hessian[np.ix_(free, free)]
        eigen_min = float(np.linalg.eigvalsh(h)[0])
        shift = max(0., -eigen_min + max(1., np.max(np.abs(h))*.01)) if eigen_min <= 0 else 0.
        step = np.zeros(2)
        try:
            step[free] = -np.linalg.solve(h+shift*np.eye(len(h)), gradient[free])
        except np.linalg.LinAlgError:
            step[free] = -gradient[free]
        radius = min(.035, .4*float(np.sqrt(r2.min())))
        length = np.linalg.norm(step)
        if length > radius:
            step *= radius/length
        accepted = False
        for _ in range(24):
            trial = np.clip(q+step, 0., 1.)
            displacement = trial-q
            directional = float(np.dot(gradient, displacement))
            if directional >= 0 or np.max(np.abs(displacement)) < 2e-16:
                break
            trial_value = float(intensity(points, [trial])[0])
            if trial_value <= value + 1e-4*directional:
                q = trial
                accepted = True
                break
            step *= .5
        if not accepted:
            break
    return q


def oracle(points, grid=39, extra=None):
    """Discover minima using local Newton basins and retain all supplied witnesses.

    This numerical search does not replace continuous-domain certification.
    Its returned minimum can never exceed the intensity at any finite supplied
    witness merely because local optimization or seed deduplication discarded it.
    """
    p=np.asarray(points,dtype=float)
    t=np.linspace(0,1,grid)
    xx, yy=np.meshgrid(t,t)
    sites=np.column_stack((xx.ravel(),yy.ravel()))
    vals=intensity(p,sites).reshape(grid,grid)
    mask=(vals==minimum_filter(vals,size=3,mode='constant',cval=np.inf))
    mask[[0,-1],:]=False; mask[:,[0,-1]]=False
    starts=[]
    if extra is not None:
        supplied=np.asarray(extra,dtype=float).reshape(-1,2)
        supplied=supplied[np.all(np.isfinite(supplied),axis=1) & np.all((supplied>=0)&(supplied<=1),axis=1)]
        starts.extend(supplied)
    else:
        supplied=np.empty((0,2))
    starts.extend(sites[mask.ravel()])
    try:
        vv=Voronoi(p).vertices
        starts.extend(vv[np.all((vv>0)&(vv<1),axis=1)])
    except QhullError:
        pass
    starts=unique_sites(starts,eps=1e-7)
    minima=[np.array([0.,0.]), np.array([0.,1.]),np.array([1.,0.]),np.array([1.,1.])]
    # Raw seeds are retained as valid upper witnesses even if a local solver fails.
    minima.extend(starts)
    for q in starts:
        if np.min(np.sum((p-q)**2,axis=1)) > 1e-24:
            minima.append(_site_newton(q,p))
    # Edge derivative bracketing complements sampled local minima. Light
    # projections are included so each geometric gap gets a separate interval.
    from scipy.optimize import brentq
    for axis in (0,1):
        for fixed in (0.,1.):
            edge_t=np.unique(np.r_[np.linspace(0,1,max(grid*3,121)),p[:,axis],supplied[:,axis]])
            qs=np.zeros((len(edge_t),2)); qs[:,axis]=edge_t; qs[:,1-axis]=fixed
            vs=intensity(p,qs)
            delta=qs[:,None,:]-p[None,:,:]
            r2=np.maximum(np.sum(delta*delta,axis=2),1e-28)
            derivatives=-2*np.sum(delta[:,:,axis]/r2**2,axis=1)
            def edge_point(t):
                z=np.zeros(2);z[axis]=t;z[1-axis]=fixed
                return z
            def fun(t):
                return float(intensity(p,[edge_point(t)])[0])
            def derivative(t):
                return site_value_grad(edge_point(t),p)[1][axis]
            indices=np.where((vs[1:-1]<=vs[:-2])&(vs[1:-1]<=vs[2:]))[0]+1
            for j in indices:
                res=minimize_scalar(fun,bounds=(edge_t[j-1],edge_t[j+1]),method='bounded',options={'xatol':1e-14})
                minima.append(edge_point(res.x))
            for j in np.where((derivatives[:-1]<0)&(derivatives[1:]>0))[0]:
                try:
                    root=brentq(derivative,edge_t[j],edge_t[j+1],xtol=5e-15,rtol=1e-14)
                    minima.append(edge_point(root))
                except (ValueError,RuntimeError):
                    pass
    # Sort before deduplicating, so the best point in each cluster always wins.
    # Preserve all input witness values regardless of closeness to another site.
    candidates=np.asarray(minima,dtype=float).reshape(-1,2)
    values=intensity(p,candidates)
    candidates=candidates[np.argsort(values)]
    minima=unique_sites(candidates,eps=1e-7)
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
        cutoff=max(vals[0]*1.25,res.x[-1]*scale*1.1)
        keep=sv<cutoff
        keep[np.argsort(sv)[:min(4,len(sv))]]=True
        sites=sites[keep]
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
