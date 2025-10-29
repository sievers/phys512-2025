import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.zeros([2,2])
v=np.zeros([2,2])
#assume m1=m2=G=1

x[0,0]=1
x[1,0]=-1
v[0,1]=0.5
v[1,1]=-0.5

v=v/2


plt.clf()
plt.plot(x[:,0],x[:,1],'*')
plt.show()

norbit=20
per_orbit=1000
dt=4*np.pi/per_orbit
nstep=norbit*per_orbit
for i in range(nstep):
    xx=x+0.5*v*dt  #guess at center position for energy calculation
    x=x+v*dt  #update positions first

    
    dr=x[0,:]-x[1,:] #vector separation of particles
    r=np.sqrt(np.sum(dr**2))
    dr_for_e=xx[0,:]-xx[1,:]
    r_for_e=np.sqrt(np.sum(dr_for_e**2))
    E=0.5*np.sum(v**2)-1/r_for_e
    print(E)

    a=dr/r**3
    v[0,:]=v[0,:]-a*dt
    v[1,:]=v[1,:]+a*dt
    plt.plot(x[:,0],x[:,1],'b*')
    plt.pause(0.00001)

