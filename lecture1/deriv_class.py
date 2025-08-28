import numpy as np
from matplotlib import pyplot as plt
plt.ion()
def f(x):
    return np.exp(x)

x=2.0
dx=1e-12

ans=np.exp(x)

deriv=(f(x+dx)-f(x))/dx
print('got ',deriv,' expected ',ans)


dx=10**np.linspace(-16,0,1000)
deriv=(f(x+dx)-f(x))/dx
plt.clf()
plt.loglog(dx,np.abs(deriv-ans))
plt.show()

deriv=(f(x+dx)-f(x-dx))/(2*dx)
plt.plot(dx,np.abs(deriv-ans))
