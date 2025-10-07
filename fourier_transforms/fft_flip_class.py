import numpy as np
from matplotlib import pyplot  as plt
plt.ion()

x=np.linspace(-5,5,1001)
y=np.exp(-0.5*(x-3)**2/0.2**2)
yft=np.fft.fft(y)
print('complex len(yft): ',len(yft))
plt.clf()
plt.plot(x,y)
plt.show()
yy=np.fft.ifft(np.conj(yft))
plt.plot(x,np.real(yy))
