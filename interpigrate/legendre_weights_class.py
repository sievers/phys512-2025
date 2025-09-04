import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def get_weights(n):
    x=np.linspace(-1,1,n+1)
    P=np.zeros([n+1,n+1])
    P[:,0]=1
    P[:,1]=x
    for nn in range(1,n):
        P[:,nn+1]=((2*nn+1)*x*P[:,nn]-nn*P[:,nn-1])/(nn+1)
    Pinv=np.linalg.inv(P)
    return Pinv[0,:]

print('second order weights are: ',get_weights(2))
ord=8
print('for order ',2,' weights are ',get_weights(ord)*ord)
print('weight sum: ',np.sum(get_weights(ord)))
