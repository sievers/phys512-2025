import numpy as np
from matplotlib import pyplot as plt
plt.ion()


def cheb_polys(x,n):
    #take an input set of x points, and evaluate
    #T_0 through T_n at them
    T=np.zeros([len(x),n+1])
    T[:,0]=1
    if n>0:
        T[:,1]=x
    if n>1:
        for i in range(1,n):
            T[:,i+1]=2*x*T[:,i]-T[:,i-1]
    return T



def cheb_fit(f,ord):
    x=np.linspace(-1,1,ord+1)
    T=cheb_polys(x,ord)
    y=f(x)
    return np.linalg.inv(T)@y

def cheb_eval(coeffs,x):
    T=cheb_polys(x,len(coeffs)-1)
    return T@coeffs

x=np.linspace(-1,1,2001)
T=cheb_polys(x,5)
plt.clf()
plt.plot(x,T)
plt.show()

coeffs=cheb_fit(np.exp,20)
print(coeffs)

x=np.linspace(-1,1,2001)
y=np.exp(x)
yy5=cheb_eval(coeffs[:7],x)
myerr=yy5-y
