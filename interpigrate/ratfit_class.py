import numpy as np
from matplotlib import pyplot as plt
plt.ion()
def f(x):
    return np.exp(-0.5*x**2)

#for a rational function fit, p(x)/(1+qq(x))=y(x)
#rewrite as p(x)-yqq(x)=y
#write as (1 x x^2... -yx -yx^2...)(p0 p1 p2... qq1 qq2...) = y(x)

def ratfit_exact(x,y,n,m):
    #n terms on top, m-1 terms on bottom
    npt=len(x)
    assert(npt==n+m-1)
    A=np.zeros([npt,npt])
    for i in range(n): #fill in top polynomial
        A[:,i]=x**i
    for j in range(1,m): #fill in bottom polynomial
        A[:,j-1+n]=-y*x**j
    #we have A*[p;qq]=y, so [p;qq]=inv(A)@y
    fitp=np.linalg.pinv(A)@y
    return fitp
def rateval(pars,x,n,m):
    top=0.0
    for i in range(n):
        top=top+x**i*pars[i]
    bot=1.0 #offset since x^0 term is 1
    for i in range(1,m):
        bot=bot+x**i*pars[i+n-1]
    return top/bot

n=4
m=8
npt=n+m-1
x=np.linspace(-1,1,npt)
y=f(x)
fitp=ratfit_exact(x,y,n,m)

xx=np.linspace(-4,4,2001)
yy=rateval(fitp,xx,n,m)
plt.clf()
plt.plot(x,y,'*')
plt.plot(xx,yy)
plt.plot(xx,f(xx))
pp=np.polyfit(x,y,npt-1)
plt.plot(xx,np.polyval(pp,xx))
plt.show()
