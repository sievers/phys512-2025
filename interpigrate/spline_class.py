import numpy as np
from scipy.interpolate import CubicSpline


def fun(x):
    return 1/np.log(x)
from matplotlib import pyplot as plt
plt.ion()

#x=np.linspace(-np.pi,np.pi,21)
x=np.linspace(2,3,41)
dx=x[1]-x[0]
#y=np.sin(x)
y=fun(x)

#y[:5]=y[5]
#y[5]=y[5]+1


spln=CubicSpline(x,y)
xx=np.linspace(x[0],x[-1],2001)
yy=spln(xx)

truth=fun(xx)
plt.clf()
plt.plot(x,y,'*')
plt.plot(xx,yy)
#plt.plot(xx,truth)
plt.show()
print('err is ',np.std(truth-yy))
