import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=int(1e6)
x=np.random.rand(n)
t=-np.log(x)

counts,bins=np.histogram(t,np.linspace(0,10,101))
bb=(bins[1:]+bins[:-1])/2
db=bb[2]-bb[1]



plt.clf()
plt.bar(bb,counts/counts.sum()/db,db)
plt.show()
plt.plot(bb,np.exp(-bb),'r')
