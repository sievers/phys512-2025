import numpy as np
import time

nmin=int(2**16)-5

niter=100
nmax=nmin+15

for n in range(nmin,nmax):
    y=np.random.randn(n)
    t1=time.time()
    for i in range(niter):
        yft=np.fft.fft(y)
    t2=time.time()
    print('average time in msec: ',(t2-t1)/niter*1000)

