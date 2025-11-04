import numpy as np
from matplotlib import pyplot as plt

alpha=-2.0

n=int(1e6)
x=np.random.rand(n)
s=x**(1/(1+alpha))

#counts,bins=np.histogram(t,np.linspace(1,11,101))
counts,bins=np.histogram(s,np.linspace(1,11,101))
bb=(bins[1:]+bins[:-1])/2
db=bb[2]-bb[1]



plt.clf()
plt.bar(bb,counts/n/db,db)
plt.show()
plt.plot(bb,bb**(alpha)*np.abs(1+alpha),'r')
