import numpy as np
import numba as nb
import time

#code to show race condition, and numba's automatic reduction for floats,
#but not numpy arrays

@nb.njit(parallel=True)
def nb_sum(vec):
    tot=0.0   
    for i in nb.prange(len(vec)):
        #numba sees you're adding into float tot, so it makes a copy per-thread
        #and silently adds them together at the end.  this is called a reduction
        #this code produces correct results
        tot=tot+vec[i] 
    return tot

@nb.njit(parallel=True)
def nb_arr_sum(vec):
    tot=np.zeros(1)
    for i in nb.prange(len(vec)):
        #numba does not do an automatic reduction if you're adding into a numpy array
        #this code will produce wrong results
        tot[0]=tot[0]+vec[i]
    return tot[0]


n=int(1e8)
vec=np.random.rand(n)
print('numpy sum: ',np.sum(vec))
print('numba sum: ',nb_sum(vec))
print('nbarr sum: ',nb_arr_sum(vec))

niter=10
t1=time.time()
for i in range(niter):
    tot=np.sum(vec)
t2=time.time()
print('numpy time: ',(t2-t1)/niter)

t1=time.time()
for i in range(niter):
    tot=nb_sum(vec)
t2=time.time()
print('numba time: ',(t2-t1)/niter)

t1=time.time()
for i in range(niter):
    tot=nb_arr_sum(vec)
t2=time.time()
print('nbarr time: ',(t2-t1)/niter)

