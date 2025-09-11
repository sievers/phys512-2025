import numpy as np
from matplotlib import pyplot as plt

def f(x,y):
    #dy/dx=f(x,y)
    #if f=-y, then dy/dx=-y, then y=exp(-x)
    return -y


n=20
h=1/n
x=np.linspace(0,1,n+1)
y=0*x
y[0]=1
for i in range(len(x)-1):
    y[i+1]=y[i]+h*f(x[i],y[i])
plt.figure(0)
plt.clf()
plt.plot(x,y)
plt.plot(x,np.exp(-x))
plt.show()
plt.figure(1)
plt.clf()
plt.plot(x,y-np.exp(-x))
plt.show()
