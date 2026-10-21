"""Tiled float64 direct-cell sum with optional native macOS BLAS.
The portable NumPy backend evaluates the identical real products.
"""

import ctypes as ct
import sys
import platform
import numpy as np

class AppleDGEMM:
    def __init__(self):
        if sys.platform!='darwin':raise OSError('Apple Accelerate requires macOS')
        self.lib=ct.CDLL('/System/Library/Frameworks/Accelerate.framework/Accelerate')
        self.symbol='cblas_dgemm$NEWLAPACK'
        self.gemm=getattr(self.lib,self.symbol)
        ptr=ct.POINTER(ct.c_double)
        self.gemm.argtypes=[ct.c_int]*6+[ct.c_double,ptr,ct.c_int,ptr,ct.c_int,ct.c_double,ptr,ct.c_int]
        self.gemm.restype=None
        self.set_threading=self.lib.BLASSetThreading;self.set_threading.argtypes=[ct.c_uint];self.set_threading.restype=ct.c_int
        self.get_threading=self.lib.BLASGetThreading;self.get_threading.argtypes=[];self.get_threading.restype=ct.c_uint

    def __call__(self,a,b):
        a=np.ascontiguousarray(a,dtype=np.float64);b=np.ascontiguousarray(b,dtype=np.float64)
        if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:raise ValueError('Expected compatible matrices')
        m,k=a.shape;n=b.shape[1]
        if max(m,n,k)>np.iinfo(np.int32).max:raise ValueError('Matrix dimensions exceed LP64 CBLAS')
        out=np.empty((m,n),dtype=np.float64)
        if not m or not n:return out
        if not k:out.fill(0.);return out
        previous=self.get_threading()
        if self.set_threading(1)!=0:raise RuntimeError('Could not select single-threaded Apple BLAS')
        ptr=ct.POINTER(ct.c_double)
        try:
            self.gemm(101,111,111,m,n,k,1.,a.ctypes.data_as(ptr),k,b.ctypes.data_as(ptr),n,
                      0.,out.ctypes.data_as(ptr),n)
        finally:self.set_threading(previous)
        return out


def direct_real(kk,kg,k0,g0,w,*,qblock=1024,matmul=None):
    from . import sgl_full as F
    e=np.exp(1j*np.outer(kk,k0))*w
    packed=np.concatenate([e.real,e.imag],axis=0)
    out=np.empty((len(kk),len(kg)),complex)
    for j in range(0,len(kg),qblock):
        b=F._j0m1(np.outer(g0,kg[j:j+qblock]))
        v=packed@b if matmul is None else matmul(packed,b)
        out.real[:,j:j+qblock]=v[:len(kk)]
        out.imag[:,j:j+qblock]=v[len(kk):]
    return out


class DirectProduct:
    def __init__(self, qblock=1024, prefer_apple=True):
        if not isinstance(qblock, int) or qblock <= 0:
            raise ValueError("qblock must be a positive integer")
        self.qblock = qblock
        self.matmul = None
        self.backend_name = "numpy_real_tiled"
        self.fallback_reason = None
        if prefer_apple and sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                backend = AppleDGEMM()
                backend(np.ones((1, 1)), np.ones((1, 1)))
            except (OSError, AttributeError, RuntimeError) as exc:
                self.fallback_reason = str(exc)
            else:
                self.matmul = backend
                self.backend_name = "apple_float64_single_thread"

    def __call__(self, kk, kg, k0, g0, w):
        return direct_real(kk, kg, k0, g0, w, qblock=self.qblock, matmul=self.matmul)
