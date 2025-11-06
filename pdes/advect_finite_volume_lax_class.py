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
nstep=int(2*n/alpha)
nn=int(1/alpha)
if nn<1:
    nn=1

for i in range(nstep):
    rr=np.zeros(n+2) #make a larger region we can fill in, one cell on each side
    rr[1:-1]=rho
    rr[0]=rho[-1] #periodic
    rr[-1]=rho[0]

    #try second order
    grad=(rr[2:]-rr[:-2])/2
    #grad=(rr[1:]-rr[:-1]) #standard upwind
    rho_new=0*rr
    #upwind derivative
    rho_lax=(rr[2:]+rr[:-2])/2
    rho_new[1:-1]=rho_lax-alpha*grad
    
    rho=rho_new[1:-1]
    if i%nn==0:
        plt.clf()
        plt.plot(rho)
        plt.ylim([-0.1,1.1])
        plt.show()
        plt.pause(0.001)

