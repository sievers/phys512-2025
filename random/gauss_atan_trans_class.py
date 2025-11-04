import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-10,10,1001)
P=np.exp(-0.5*x**2)
y=np.arctan(x)
PP=np.exp(-0.5*np.tan(y)**2)/np.cos(y)**2
plt.figure(1)
plt.clf()
plt.plot(y,PP)
plt.show()

height=1.1*np.max(PP)
n=int(1e7)
vals_raw=np.pi*(np.random.rand(n)-0.5) #random between -pi/2 and pi/2

accept=(np.random.rand(n)<(np.exp(-0.5*np.tan(vals_raw)**2)/np.cos(vals_raw)**2/height))
yvals=vals_raw[accept]
vals=np.tan(yvals)
plt.figure(2)
bins=np.linspace(-5,5,501)
aa,bb=np.histogram(vals,bins)
b=(bb[1:]+bb[:-1])/2
db=b[2]-b[1]

aa_norm=aa/aa.sum()/db
plt.clf()
plt.bar(b,aa_norm,db)
plt.show()
plt.plot(b,np.exp(-0.5*b**2)/(np.sqrt(2*np.pi)),'r')
