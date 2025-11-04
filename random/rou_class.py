import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=int(1e7)
u=np.random.rand(n)
vmax=0.85
v=(np.random.rand(n)-0.5)*2*vmax

#plt.clf()
#plt.plot(u,v,'.')
#plt.show()

accept=(u<np.sqrt(np.exp(-0.5*(v/u)**2)))
print('accept fraction is ',np.mean(accept))
#reject=(u>np.sqrt(np.exp(-0.5*(v/u)**2)))
vals=(v/u)[accept]

#plt.clf()
#plt.plot(u[accept],v[accept],'.')
#plt.plot(u[reject],v[reject],'r.')
#plt.show()

bins=np.linspace(-5,5,501)
aa,bb=np.histogram(vals,bins)
b=(bb[1:]+bb[:-1])/2
db=b[2]-b[1]

aa_norm=aa/aa.sum()/db
plt.clf()
plt.bar(b,aa_norm,db)
plt.show()
plt.plot(b,np.exp(-0.5*b**2)/(np.sqrt(2*np.pi)),'r')

