import numpy as np
from matplotlib import pyplot as plt

plt.ion()

x=np.linspace(-1,1,2001)

ord=5
P=np.zeros([len(x),ord+1])
P[:,0]=1.0
P[:,1]=x
for n in range(1,ord):
    P[:,n+1]=((2*n+1)*x*P[:,n]-n*P[:,n-1])/(n+1)

plt.clf()
plt.plot(x,P)
plt.show()
overlap=P.T@P/len(x)
print(overlap)
