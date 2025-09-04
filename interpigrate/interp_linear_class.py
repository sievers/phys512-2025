import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def f(x):
    return 1/x


x=np.linspace(1,2,21)
y=f(x)
ans=np.log(x[-1])-np.log(x[0]) #analytic answer for log(x)

dx=x[1]-x[0] #the spacing of our points
myans=(0.5*(y[0]+y[-1])+np.sum(y[1:-1]))*dx
print('got ',myans,' expected ',ans,' error: ',myans-ans)
