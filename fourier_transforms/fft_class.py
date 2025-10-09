import numpy as np
import time


def fft(y):
    N=len(y)
    if N==1:
        return y
    yft_even=fft(y[::2])
    yft_odd=fft(y[1::2])
    NN=N//2
    twid=np.exp(-2J*np.pi*np.arange(NN)/N)
    return np.concatenate((yft_even+twid*yft_odd,yft_even-twid*yft_odd))


n=1024*256*2
y=np.random.randn(n)
t1=time.time()
yft=np.fft.fft(y)
t2=time.time()
myft=fft(y)
t3=time.time()
print('error is ',np.std(yft-myft))
print('times for numpy/ours are ',t2-t1,t3-t2)
