import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def f(x):
    return 1/np.log(x)

x=np.linspace(2,3,11)
y=f(x)
xx=np.linspace(x[1],x[-2]-1e-6,1001)
yy=np.empty(len(xx))
for i in range(len(xx)):
    ind=np.max(np.where(xx[i]>=x)[0]) #this is the line I had wrong
    pp=np.polyfit(x[ind-1:ind+3],y[ind-1:ind+3],3)
    yy[i]=np.polyval(pp,xx[i])
truth=f(xx)
print('average error: ',np.std(yy-truth))
plt.clf()
plt.plot(x,y,'*')
plt.plot(xx,truth)
plt.plot(xx,yy)
plt.show()
