import numpy as np
from scipy import fft
import time

n=512

a=np.random.randn(n,n,n)

for i in range(5):
    t1=time.time()
    b=np.fft.rfftn(a)
    t2=time.time()
    print('numpy time: ',t2-t1)

for i in range(5):
    t1=time.time()
    b=fft.rfftn(a)
    t2=time.time()
    print('scipy time: ',t2-t1)

for i in range(5):
    t1=time.time()
    b=fft.rfftn(a,workers=12)
    t2=time.time()
    print('scipy parallel time: ',t2-t1)
