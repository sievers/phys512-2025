import numpy as np
from matplotlib import pyplot as plt
plt.ion()
#first step - generate lorentzians

n=100000
nums=np.random.rand(n)
#cdf is arctan, so inverse(cdf) is tangent
vals=np.tan( (nums-1/2)*np.pi)  #(nums-1/2)*pi gives me things on (-pi/2,pi/2)

bins=np.linspace(-5,5,101)
a,b=np.histogram(vals,bins)
bb=(bins[:-1]+bins[1:])/2
plt.clf()
plt.plot(bb,a/a.max())
plt.plot(bb,1/(1+bb**2))
plt.show()

plt.plot(bb,np.exp(-0.5*bb**2))

#vals is our random  numbers
accept=.8*np.exp(-0.5*vals**2)/(1/(1+vals**2))

gdevs=vals[np.random.rand(n)<accept]

#plt.figure(2)
a2,b2=np.histogram(gdevs,bins)
plt.plot(bb,a2/a2.max())
