import numpy as np
from matplotlib import pyplot as plt
plt.ion()

def get_kernel(npix,n):
    x=np.arange(n)
    x[x>n//2]=x[x>n//2]-n
    #can also do this with n*np.fft.fftfreq(n)
    mygauss=np.exp(-0.5*(x/npix)**2)  #1-d gaussian
    return np.outer(mygauss,mygauss)



cloud=plt.imread('cloudberry.png')
arts=plt.imread('mcgill_arts.jpeg')


#do some resizing/cropping to make the images the same size
#arts=arts[::2,::2,0]
cloud=cloud[:,:,0]
#i0=(arts.shape[0]-cloud.shape[0])//2
#i1=(arts.shape[1]-cloud.shape[1])//2
#arts=arts[i0:i0+cloud.shape[0],i1:i1+cloud.shape[1]]

dn=cloud.shape[1]-cloud.shape[0]
cloud=cloud[:,dn//2:-dn//2]

cloudft=np.fft.rfft2(cloud)

kernel=get_kernel(3,cloud.shape[0])
kernelft=np.fft.rfft2(kernel)
cloudft_smooth=cloudft*kernelft
cloud_smooth=np.fft.irfft2(cloudft_smooth)
plt.figure(1)
plt.clf();
plt.imshow(cloud)
plt.show()

plt.figure(2)
plt.clf();
plt.imshow(cloud_smooth)
plt.show()


plt.figure(3)
cloudsmoothft=np.fft.rfft2(cloud_smooth)
cloud_back=np.fft.irfft2(cloudsmoothft/kernelft)
plt.clf()
plt.imshow(cloud_back)
plt.show()
