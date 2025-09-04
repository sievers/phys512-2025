import numpy as np

def fun(x):
    return 1/np.log(x)
from matplotlib import pyplot as plt
plt.ion()

#x=np.linspace(-np.pi,np.pi,21)
x=np.linspace(2,3,21)
dx=x[1]-x[0]
#y=np.sin(x)
y=fun(x)

#xx=np.linspace(-np.pi,np.pi,1001)
#yy=np.sin(xx)
xx=np.linspace(x[1],x[-1],1001)
yy=fun(xx)
plt.clf()
plt.plot(xx,yy)
plt.plot(x,y,'.')
plt.show()

myind=(xx-x[0])/dx

#plt.clf()
#plt.plot(xx,myind)
#plt.show()
yy_interp=0*yy
for i in range(len(xx)):
    #check which bin we live in
    ii=int(myind[i])
    #make sure our bin isn't at the edge
    if (ii>=1):
        if (ii<len(x)-2):
            #find our neighboring x and y values
            myx=x[ii-1:ii+3]
            myy=y[ii-1:ii+3]
            #do the fit
            fitp=np.polyfit(myx,myy,3)
            #evaluate
            yy_interp[i]=np.polyval(fitp,xx[i])
plt.plot(xx,yy_interp)
