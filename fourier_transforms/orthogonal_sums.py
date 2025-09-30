import numpy as np
np.set_printoptions(precision=3)
N=10
x=np.arange(N)
for k in range(N):
    vec=np.exp(-2J*np.pi*k*x/N)
    print('k=',k,f' has sum {np.sum(vec):.3g}')
#={k:2d}')
    #print('k=',k,' has sum ',np.sum(vec))

