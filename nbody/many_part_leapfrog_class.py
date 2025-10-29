import numpy as np
from matplotlib import pyplot as plt
plt.ion()

class Parts:
    def __init__(self,x,v,soft=0.001):
        self.x=x.copy()
        self.v=v.copy()
        self.soft=0.001
    def get_forces(self):
        forces=0*self.x #initialize array with zeros of the right shape
        npart=self.x.shape[0]
        for i in range(npart):
            for j in range(npart):
                dr=self.x[i,:]-self.x[j,:]
                rsqr=np.sum(dr**2)+self.soft**2
                #r=np.sqrt(rsqr)
                forces[i,:]=forces[i,:]-dr/(rsqr**(1.5))
                #forces[i,:]=forces[i,:]+dr/(r**3)
        return forces
    def update(self,dt):
        self.x=self.x+dt*self.v
        forces=self.get_forces()
        self.v=self.v+dt*forces





if False:        
    x=np.zeros([2,2])
    v=np.zeros([2,2])
    #assume m1=m2=G=1
    x[0,0]=1
    x[1,0]=-1
    v[0,1]=0.5
    v[1,1]=-0.5
    v=v/2
else:
    npart=1000
    x=np.random.randn(npart,2)
    v=0*x

parts=Parts(x,v)

norbit=20
per_orbit=1000
dt=4*np.pi/per_orbit
nstep=norbit*per_orbit
plt.clf()
plt.plot(x[:,0],x[:,1],'b*')
plt.show()
for i in range(nstep):
    parts.update(dt)
    plt.clf()
    plt.plot(parts.x[:,0],parts.x[:,1],'b*')
    plt.xlim([-2,2])
    plt.ylim([-2,2])
    plt.pause(0.001)
