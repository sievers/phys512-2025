import numpy as np
from matplotlib import pyplot as plt
plt.ion()


x=np.arange(1000)
template=np.exp(-0.1*x) #our instrumental response will be an exponential decay

#put in a true signal in the middle, at position x0 with height amp
x0=500
amp=10
y_true=0*x
y_true[x0]=amp

#get the noiseless data after convolving with instrument response
y_conv=np.fft.irfft(np.fft.rfft(template)*np.fft.rfft(y_true))  
y=y_conv+np.random.randn(len(x)) #and add noise to get our data 

plt.figure(1)
plt.clf()
plt.plot(x,y)
plt.show()

#first, find the matched-filter output
#mf=np.fft.irfft(np.fft.rfft(y)*np.conj(np.fft.rfft(template)))/np.sum(template**2)
mf=np.fft.irfft(np.fft.rfft(y)*np.conj(np.fft.rfft(template)))/np.sum(template**2)

#now deconvolve the data so we can compare to mf
deconv=np.fft.irfft(np.fft.rfft(y)/np.fft.rfft(template))

plt.figure(2)
plt.clf()
plt.plot(x,deconv)
plt.plot(x,mf)
plt.plot(x,y_true,'.')
plt.legend(['Deconvolved','Matched Filter','Truth'])
plt.show()

correction=np.median(np.abs(np.random.randn(1000000))) #find conversion between median and sigma
mf_noise=np.median(np.abs(mf))/correction
conv_noise=np.median(np.abs(deconv))/correction

print('mf/deconv peaks are ',np.max(mf),np.max(deconv))
print('mf/deconv noises are ',mf_noise,conv_noise)
                     
