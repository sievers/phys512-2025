import numpy as np
import time

n=5000
x=np.random.randn(n,n).astype('float64')
t1=time.time()
y=x@x
t2=time.time()
print('double took ',t2-t1,' seconds.')
xx=x.astype('float32')
t1=time.time()
yy=xx@xx
t2=time.time()
print('single took ',t2-t1,' seconds.')
