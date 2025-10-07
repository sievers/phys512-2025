import numpy as np
from matplotlib import pyplot as plt
plt.ion()

N=1000
y_true=(1+np.random.randn(N))**5
plt.clf();plt.plot(y_true);plt.show()

x=np.arange(N)
kernel=np.exp(-x/1)
kernel=kernel/kernel.sum()

y_measured=np.fft.irfft(np.fft.rfft(kernel)*np.fft.rfft(y_true))
plt.plot(y_measured)
