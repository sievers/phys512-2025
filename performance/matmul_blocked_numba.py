import numpy as np
import numba as nb
import time

@nb.njit
def matmul_old(a,b,c):
    n=a.shape[0]
    m=b.shape[1]
    k=a.shape[1]
    for i in range(n):
        for j in range (m):
            c[i,j]=0
            for kk in range(k):
                c[i,j]=c[i,j]+a[i,kk]*b[kk,j]

@nb.njit
def matmul_block(a,b,c,n):
    for i in range(n):
        for j in range (n):
            #c[i,j]=0
            for kk in range(n):
                c[i,j]=c[i,j]+a[i,kk]*b[kk,j]

@nb.njit(parallel=True)
def matmul(a,b,c,bs=32):
    c[:]=0
    n=a.shape[0]
    nbl=n//bs
    for i in nb.prange(nbl):
        atmp=np.zeros((bs,bs))
        btmp=np.zeros((bs,bs))
        ctmp=np.zeros((bs,bs))

        for j in np.arange(nbl):
            ctmp[:]=0
            for k in range(nbl):
                atmp[:,:]=a[i*bs:(i+1)*bs,k*bs:(k+1)*bs]
                btmp[:,:]=b[k*bs:(k+1)*bs,j*bs:(j+1)*bs]
                matmul_block(atmp,btmp,ctmp,bs)
            c[i*bs:(i+1)*bs,j*bs:(j+1)*bs]=ctmp


    
n=256
bs=32
a=np.random.randn(n,n)
b=np.random.randn(n,n)
c=np.zeros([n,n])
matmul(a,b,c,bs)
#matmul_block(a,b,c,n)
print('error: ',np.std(c-a@b))

n=2048
a=np.random.randn(n,n)
b=np.random.randn(n,n)
c=np.empty([n,n])
matmul(a,b,c)
t1=time.time()
matmul(a,b,c,bs)
t2=time.time()
print('time: ',t2-t1)
flops=2*n**3/(t2-t1)
print('flop rate: ',flops/1e9)
print('big error: ',np.std(c-a@b))
