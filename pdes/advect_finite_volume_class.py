import numpy as np
from matplotlib import pyplot as plt
plt.ion()


n=300
rho=np.zeros(n)
rho[n//3:(2*n//3)]=1

plt.clf()
plt.plot(rho)
plt.show()

alpha=0.99
nstep=int(n/alpha)
nn=int(1/alpha)
if nn<1:
    nn=1

for i in range(nstep):
    rho_new=0*rho
    #upwind derivative
    rho_new[1:]=rho[1:]*(1-alpha)+alpha*rho[:-1]
    #try downwind derivatve
    #rho_new[:-1]=rho[:-1] -alpha*rho[1:]+alpha*rho[:-1] 
    rho=rho_new
    if i%nn==0:
        plt.clf()
        plt.plot(rho)
        plt.show()
        plt.pause(0.001)

