import numpy as np
from matplotlib import pyplot as plt

plt.ion()

cloud=plt.imread('cloudberry.png')
arts=plt.imread('mcgill_arts.jpeg')


#do some resizing/cropping to make the images the same size
arts=arts[::2,::2,0]
cloud=cloud[:,:,0]
i0=(arts.shape[0]-cloud.shape[0])//2
i1=(arts.shape[1]-cloud.shape[1])//2
arts=arts[i0:i0+cloud.shape[0],i1:i1+cloud.shape[1]]

artsft=np.fft.rfft2(arts)
cloudft=np.fft.rfft2(cloud)

im1=np.fft.irfft2(np.abs(artsft)*np.exp(1J*np.angle(cloudft)))
im2=np.fft.irfft2(np.abs(cloudft)*np.exp(1J*np.angle(artsft)))
