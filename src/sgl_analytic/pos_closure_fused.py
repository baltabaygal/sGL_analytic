"""Structured evaluation of exact-group VP rates.

Each sampled VP rate is affine on one contiguous interval of latent states.
Accumulate its slope/intercept at interval boundaries, then prefix-sum over
states. This combines the original group sum and rate matrix product without
materializing a (number_of_groups, number_of_points) intermediate.
The latent nodes, transition matrices, Fourier grids and recursion are unchanged.
"""
from __future__ import annotations

from time import perf_counter

import numba
import numpy as np
from scipy.special import j0


def rate_intervals(d, u):
    """Linear D uses a*u+b; clipped rates use a*(u-u_anchor)+b, minus 1.

    The clipped anchor is the smallest positive rate (left for increasing
    rates, right for decreasing rates), avoiding large-number cancellation.
    """
    d=np.asarray(d,float);u=np.asarray(u,float)
    rate=d+1.
    active=rate>0
    count=active.sum(axis=0)
    if np.any(count==0):raise ValueError('VP rate has no active state')
    lo=active.argmax(axis=0).astype(np.int64)
    hi=(len(u)-1-active[::-1].argmax(axis=0)).astype(np.int64)
    if np.any(hi-lo+1!=count):raise ValueError('VP active states must form an interval')
    all_active=count==len(u)
    c=np.where(all_active,0.,-1.)
    col=np.arange(d.shape[1])
    # Fit D directly for the all-active linear branch, avoiding cancellation
    # between a large constant group sum and its identical rate intercept.
    first=np.where(all_active,d[lo,col],rate[lo,col])
    last=np.where(all_active,d[hi,col],rate[hi,col])
    span=u[hi]-u[lo]
    a=np.divide(last-first,span,out=np.zeros_like(first),where=span!=0)
    b=np.where(all_active,.5*(first+last-a*(u[lo]+u[hi])),np.where(a>=0,first,last))
    anchor=np.where(all_active,0.,np.where(a>=0,u[lo],u[hi]))
    if np.any((~all_active)&(lo!=0)&(hi!=len(u)-1)):
        raise ValueError('Clipped VP support must reach one end of the state grid')
    model=c[None,:]+np.where(active,(u[:,None]-anchor)*a+b,0.)
    scaled=float(np.max(np.abs(model-d)/np.maximum(1.,np.abs(d))))
    if scaled>2e-12:raise ValueError(f'Rate reconstruction exceeds accuracy bound: {scaled}')
    return a,b,c,lo,hi,scaled


@numba.njit(parallel=True,cache=True)
def _fused_groups(er,ei,J,Jm1,cu,cw,gstart,sh_g,aa,bb,cc,lo,hi,u,zr,zi):
    nka,nq=er.shape[1],J.shape[1]
    ns=u.size;points=nka*nq
    for s in numba.prange(sh_g.size-1):
        ar=np.zeros((ns,points));ai=np.zeros((ns,points))
        br=np.zeros((ns,points));bi=np.zeros((ns,points))
        nar=np.zeros((ns,points));nai=np.zeros((ns,points))
        nbr=np.zeros((ns,points));nbi=np.zeros((ns,points))
        cr=np.zeros(points);ci=np.zeros(points)
        la=np.zeros(points);lai=np.zeros(points);lb=np.zeros(points);lbi=np.zeros(points)
        accr=np.empty((nka,nq));acci=np.empty((nka,nq));S=np.empty(nq)
        wr=np.empty(nka);wi=np.empty(nka)
        for g in range(sh_g[s],sh_g[s+1]):
            if aa[g]==0. and bb[g]==0. and cc[g]==0.:continue
            accr[:,:]=0.;acci[:,:]=0.;S[:]=0.
            for cell in range(gstart[g],gstart[g+1]):
                idx=cu[cell];w=cw[cell]
                for q in range(nq):S[q]+=w*Jm1[idx,q]
                for k in range(nka):
                    wr[k]=w*er[idx,k];wi[k]=w*ei[idx,k]
                for k in range(nka):
                    x=wr[k];y=wi[k]
                    for q in range(nq):
                        j=J[idx,q]
                        accr[k,q]+=x*j;acci[k,q]+=y*j
            for k in range(nka):
                for q in range(nq):
                    p=k*nq+q;x=accr[k,q]+S[q];y=acci[k,q]
                    ax=aa[g]*x;ay=aa[g]*y;bx=bb[g]*x;by=bb[g]*y
                    if cc[g]==0.:
                        la[p]+=ax;lai[p]+=ay;lb[p]+=bx;lbi[p]+=by
                    elif hi[g]==ns-1:
                        left=lo[g]
                        ar[left,p]+=ax;ai[left,p]+=ay;br[left,p]+=bx;bi[left,p]+=by
                        cr[p]-=x;ci[p]-=y
                    else:
                        right=hi[g]
                        nar[right,p]+=ax;nai[right,p]+=ay;nbr[right,p]+=bx;nbi[right,p]+=by
                        cr[p]-=x;ci[p]-=y
        for p in range(points):
            ax=0.;ay=0.;bx=0.;by=0.
            for j in range(ns):
                if j>0:
                    du=u[j]-u[j-1];bx+=du*ax;by+=du*ay
                bx+=br[j,p];by+=bi[j,p];ax+=ar[j,p];ay+=ai[j,p]
                zr[s,j,p]=u[j]*la[p]+lb[p]+bx+cr[p]
                zi[s,j,p]=u[j]*lai[p]+lbi[p]+by+ci[p]
            ax=0.;ay=0.;bx=0.;by=0.
            for j in range(ns-1,-1,-1):
                if j<ns-1:
                    du=u[j]-u[j+1];bx+=du*ax;by+=du*ay
                bx+=nbr[j,p];by+=nbi[j,p];ax+=nar[j,p];ay+=nai[j,p]
                zr[s,j,p]+=bx;zi[s,j,p]+=by


class FusedVP:
    def __init__(self,pos):
        start=perf_counter()
        if pos.kind!='vp' or pos.h_a:
            raise ValueError('Structured kernel requires exact-group VP rates')
        self.pos=pos
        parts=[rate_intervals(d,pos.u) for d in pos._B['DT']]
        self.a,self.b,self.c,self.lo,self.hi=[np.ascontiguousarray(np.concatenate([p[j] for p in parts])) for j in range(5)]
        self.max_rate_reconstruction_error=max(p[-1] for p in parts)
        self.setup_seconds=perf_counter()-start
        self.calls=0;self.seconds=0.

    def lnR_blocked(self,kv,qv,nkb=16,nqb=128,max_bytes=256e6):
        from . import pos_closure as PC
        t=perf_counter();pos=self.pos;B=pos._B
        kv,qv=np.atleast_1d(np.asarray(kv,float)),np.atleast_1d(np.asarray(qv,float))
        nsh,ns=len(B['DT']),len(pos.w)
        # This bound concerns output state buffers, not total process memory.
        while nkb*nqb>1 and nsh*ns*nkb*nqb*16>max_bytes:
            if nkb>=nqb:nkb=max(1,nkb//2)
            else:nqb=max(1,nqb//2)
        out=np.empty((len(kv),len(qv)),complex)
        for i in range(0,len(kv),nkb):
            kb=kv[i:i+nkb];e=np.expm1(1j*np.outer(pos.ku,kb))
            er,ei=np.ascontiguousarray(e.real),np.ascontiguousarray(e.imag)
            for j in range(0,len(qv),nqb):
                qb=qv[j:j+nqb];J=np.ascontiguousarray(j0(np.outer(pos.gu,qb)))
                zr=np.empty((nsh,ns,len(kb)*len(qb)));zi=np.empty_like(zr)
                _fused_groups(er,ei,J,J-1.,B['cu'],B['cw'],B['gstart'],B['sh_g'],
                              self.a,self.b,self.c,self.lo,self.hi,pos.u,zr,zi)
                real,imag=PC._chain_recursion(zr,zi,pos._Karr,pos.w)
                out[i:i+len(kb),j:j+len(qb)]=(real+1j*imag).reshape(len(kb),len(qb))
        self.calls+=1;self.seconds+=perf_counter()-t
        return out
