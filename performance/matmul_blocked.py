import numpy as np
import numba as nb
import time

@nb.njit
def matmul(a,b,c):
    n=a.shape[0]
    m=b.shape[1]
    k=a.shape[1]
    for i in range(n):
        for j in range (m):
            c[i,j]=0
            for kk in range(k):
                c[i,j]=c[i,j]+a[i,kk]*b[kk,j]

n=256
#bs=
a=np.random.randn(n,n)
b=np.random.randn(n,n)
c=np.empty([n,n])
matmul(a,b,c)
print('error: ',np.std(c-a@b))

n=1024
a=np.random.randn(n,n)
b=np.random.randn(n,n)
c=np.empty([n,n])
t1=time.time()
matmul(a,b,c)
t2=time.time()
print('time: ',t2-t1)
flops=2*n**3/(t2-t1)
print('flop rate: ',flops/1e9)

bs=32
nb=n//bs
tmp=np.empty([bs,bs])
t1=time.time()
c[:,:]=0
for i in range(nb):
    for j in range(nb):
        for k in range(nb):
            ablock=a[i*bs:(i+1)*bs,k*bs:(k+1)*bs]
            bblock=b[k*bs:(k+1)*bs,j*bs:(j+1)*bs]
            matmul(ablock,bblock,tmp)
            c[i*bs:(i+1)*bs,j*bs:(j+1)*bs]=tmp+c[i*bs:(i+1)*bs,j*bs:(j+1)*bs]
t2=time.time()
print('blocked time: ',t2-t1)
