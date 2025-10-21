import numpy as np
from matplotlib import pyplot as plt
plt.ion()

x=np.arange(1000)
tau=30 
kernel=np.exp(-x/tau) #instrument response is exponential decay

x0=500
amp=4
y_true=0*x
y_true[x0]=amp
y_noiseless=np.fft.irfft(np.fft.rfft(kernel)*np.fft.rfft(y_true))
dat=y_noiseless+np.random.randn(len(x))

plt.clf()
plt.plot(x,y_true)
plt.plot(x,y_noiseless)
plt.plot(x,dat)
plt.show()

#matched filter
dft=np.fft.rfft(dat)
kft=np.fft.rfft(kernel)
bot=np.sum(kernel**2)
mf=np.fft.irfft(dft*np.conj(kft))/bot
deconv=np.fft.irfft(dft/kft)
plt.plot(x,deconv)
plt.plot(x,mf)
print('noises: ',np.median(np.abs(mf)),np.median(np.abs(deconv)))

