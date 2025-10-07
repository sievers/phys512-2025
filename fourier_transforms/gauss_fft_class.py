import numpy as np
from matplotlib import pyplot  as plt
plt.ion()

x=np.linspace(-5,5,1001)
y=np.exp(-0.5*x**2/0.5**2)
yft=np.fft.fft(y)
plt.clf()
plt.plot(x,y)
plt.plot(x,np.fft.fftshift(np.abs(yft))/np.abs(yft[0]))
plt.show()
