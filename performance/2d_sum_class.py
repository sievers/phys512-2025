import numpy as np
import time
import numba as nb

@nb.njit(parallel=True)
def sum_row(mat):
    n=mat.shape[0]
    m=mat.shape[1]
    out=np.zeros(n)
    for i in nb.prange(n):
        for j in np.arange(m):
            out[i]=out[i]+mat[i,j]
    return out
@nb.njit(parallel=True)
def sum_col(mat):
    n=mat.shape[0]
    m=mat.shape[1]
    out=np.zeros(m)
    for j in nb.prange(m):
        for i in np.arange(n):
            out[j]=out[j]+mat[i,j]
    return out

a=np.random.randn(5000,5000)
rsum=sum_row(a)
csum=sum_col(a)

for i in range(20):
    t1=time.time()
    rsum=sum_row(a.T)
    t2=time.time()
    csum=sum_col(a.T)
    t3=time.time()
    print('row/col sum times are: ',(t2-t1)/1e-3,(t3-t2)/1e-3)
