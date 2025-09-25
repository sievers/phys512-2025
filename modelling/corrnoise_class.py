import numpy as np
from matplotlib import pyplot as plt
plt.ion()


ndata=1000
N=np.zeros([ndata,ndata])

#we'll decree that N_ij = exp(-0.5*(i-j)^2/width^2)
width=10
for i in range(ndata):
    for j in range(ndata):
        N[i,j]=np.exp(-0.5*(i-j)**2/width**2)
plt.clf()
plt.imshow(N)
plt.show()
e,v=np.linalg.eigh(N)
e[e<0]=0 #clean up roundoff error

d_uncorr=np.sqrt(e)*np.random.randn(len(e))  #this uncorrelated noisy data
d=v@d_uncorr

plt.clf()
plt.plot(d)
plt.show()
