import numpy as np
from matplotlib import pyplot as plt
plt.ion()

n=int(1e5)
u=np.random.rand(n)*1.12
vmax=1.8
v=(np.random.rand(n)-0.5)*2*vmax

#plt.clf()
#plt.plot(u,v,'.')
#plt.show()

#let's do P(x) propto (1+x^2)*gauss
accept=(u<np.sqrt((1+(v/u)**2)*np.exp(-0.5*(v/u)**2)))
print('accept fraction is ',np.mean(accept))
reject=(u>np.sqrt((1+(v/u)**2)*np.exp(-0.5*(v/u)**2)))
vals=(v/u)[accept]

plt.clf()
plt.plot(u[accept],v[accept],'.')
plt.plot(u[reject],v[reject],'r.')
plt.show()
assert(1==0)


bins=np.linspace(-5,5,501)
aa,bb=np.histogram(vals,bins)
b=(bb[1:]+bb[:-1])/2
db=b[2]-b[1]

aa_norm=aa/aa.sum()/db
plt.clf()
plt.bar(b,aa_norm,db)
plt.show()
myy=(1+b**2)*np.exp(-0.5*b**2)
plt.plot(b,myy/myy.sum()*aa_norm.sum(),'r')

