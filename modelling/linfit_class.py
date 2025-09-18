import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.linspace(-3,3,3001)
y_true=(x**4+3*x**3-3*x**2+2.5*x-1)/20

plt.clf()
plt.plot(x,y_true)
plt.show()

y=y_true+np.random.randn(len(x))
plt.plot(x,y,'.')

order=5
A=np.zeros([len(y),order])
#we want A*m to give me polynomial in x
#so A[:,0]=1
#A[:,1]=x
#A[:,2]=x**2

#A*(x**0 coeff, x**1 coeff, x**2 coeff...)
for i in range(order):
    A[:,i]=x**i

ptrue=np.asarray([-1,2.5,-3,3,1])/20
yy=A@ptrue #should be true model
plt.plot(x,yy)

lhs=A.T@A
rhs=A.T@y
fitp=np.linalg.inv(lhs)@rhs
print('ptrue: ',ptrue)
print('fitp: ',fitp)

y_fit=A@fitp  #our best guess for what the data should have been
plt.plot(x,y_fit)
