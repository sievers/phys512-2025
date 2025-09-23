import numpy as np
from matplotlib import pyplot as plt
plt.ion()

np.random.seed(5)

x=np.linspace(-1,1,3001)
y_true=(7*x**4+13*x**3-31.2*x**2+2.5*x-1)/10

plt.clf()
plt.plot(x,y_true)
plt.show()

y=y_true+np.random.randn(len(x))
plt.plot(x,y,'.')

order=250
#A=np.zeros([len(y),order])
#we want A*m to give me polynomial in x
#so A[:,0]=1
#A[:,1]=x
#A[:,2]=x**2

#A*(x**0 coeff, x**1 coeff, x**2 coeff...)
#for i in range(order):
#    A[:,i]=x**i

A=np.polynomial.legendre.legvander(x,order)

#ptrue=np.asarray([-1,2.5,-3,3,1])/20
#yy=A@ptrue #should be true model
#plt.plot(x,yy)

#lhs=A.T@A
#rhs=A.T@y
#u,s,v=np.linalg.svd(lhs,0)
#sinv=1/s
#mask=s<(1e-8*s.max()) #find small-ish singular valus
#sinv[mask]=0 #zap 'em out in the inverse!
#lhs_inv=v.T@(np.diag(sinv))@u.T
#lhs_inv=np.linalg.pinv(lhs)


lhs=A.T@A
rhs=A.T@y
fitp=np.linalg.inv(lhs)@rhs

#fitp=np.linalg.inv(lhs)@rhs
#fitp=lhs_inv@rhs
#print('ptrue: ',ptrue)
#print('fitp: ',fitp)




y_fit=A@fitp  #our best guess for what the data should have been
plt.plot(x,y_fit)
print('chisq is ',np.sum((y_fit-y)**2))

e,v=np.linalg.eigh(lhs)
print('rcond is: ',e.max()/e.min())
