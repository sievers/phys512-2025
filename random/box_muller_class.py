import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=int(1e6)
rsqr=np.random.rand(n)
th=np.random.rand(n)*2*np.pi

rr=np.sqrt(-2*np.log(rsqr))
v1=np.cos(th)*rr
v2=np.sin(th)*rr

counts,bins=np.histogram(v1,np.linspace(-5,5,101))
bb=(bins[1:]+bins[:-1])/2
db=bb[2]-bb[1]
plt.clf()
plt.bar(bb,counts/counts.sum()/db,db)
plt.show()
plt.plot(bb,np.exp(-0.5*bb**2)/np.sqrt(2*np.pi),'r')

                        
