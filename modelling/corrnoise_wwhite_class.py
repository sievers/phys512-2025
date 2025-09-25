import numpy as np
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt
import time
plt.ion()


ndata=1000
N=np.zeros([ndata,ndata])

#we'll decree that N_ij = exp(-0.5*(i-j)^2/width^2)
width=3
Nvec=np.exp(-0.5*(np.arange(ndata))**2/width**2)

white=0.001
#N=toeplitz(Nvec)+np.eye(ndata)*white**2
Nvec[0]=Nvec[0]+white**2
N=toeplitz(Nvec)

plt.figure(1)
plt.clf()
plt.imshow(N)
plt.show()
e,v=np.linalg.eigh(N)
e[e<0]=0 #clean up roundoff error

d_uncorr=np.sqrt(e)*np.random.randn(len(e))  #this uncorrelated noisy data
d=v@d_uncorr
plt.figure(2)
plt.clf()
#plt.plot(d)
plt.show()

xvec=np.arange(ndata)
xvec=xvec-xvec.mean()
sig_width=150
amp=1
A=np.exp(-0.5*xvec**2/sig_width**2)
d_use=d+A*amp
plt.plot(d_use)

#get best-fit amp.  A^T N^-1 A m = A^T N^-1 d
Ninv=np.linalg.inv(N)
lhs=A.T@Ninv@A
rhs=A.T@Ninv@d_use
fitp=rhs/lhs #np.linalg.inv(lhs)@rhs
err=1/np.sqrt(lhs)  #parameter covariance = (A^T N^-1 A)^-1
print('best-fit amplitude: ',fitp,' with error ',err)
plt.plot(A*amp)
