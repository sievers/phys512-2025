import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=2*np.random.rand(1000000,2)-1 #uniform between -1 and 1
rsqr=np.sum(x**2,axis=1)
mask=rsqr<1 #find all point pairs in the unit circle
xx=x[mask]
rr=rsqr[mask]
th=np.arctan2(xx[:,0],xx[:,1])
z=np.sqrt(-2*np.log(rr))
v1=np.cos(th)*z
v2=np.sin(th)*z
a,b=np.histogram(v1,np.linspace(-5,5,201))
bb=0.5*(b[:-1]+b[1:])

plt.clf()
plt.bar(bb,a/a.sum()/(bb[2]-bb[1]),0.05)
plt.show()
plt.plot(bb,np.exp(-0.5*bb**2)/np.sqrt(2*np.pi),'r')
