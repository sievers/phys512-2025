import numpy as np
from matplotlib import pyplot as plt
plt.ion()


n=300
rho=np.zeros(n)
rho[n//3:(2*n//3)]=1

plt.clf()
plt.plot(rho)
plt.show()

alpha=0.2
nstep=int(30*n/alpha)
nn=int(1/alpha)
if nn<1:
    nn=1

for i in range(nstep):
    rr=np.zeros(n+1) #make a larger region we can fill in
    rr[1:]=rho
    rr[0]=rho[-1] #periodic
    rho_new=0*rr
    #upwind derivative
    rho_new[1:]=rr[1:]*(1-alpha)+alpha*rr[:-1]
    
    #try downwind derivatve
    #rho_new[:-1]=rho[:-1] -alpha*rho[1:]+alpha*rho[:-1] 
    rho=rho_new[1:]
    if i%nn==0:
        plt.clf()
        plt.plot(rho)
        plt.ylim([-0.1,1.1])
        plt.show()
        plt.pause(0.001)

