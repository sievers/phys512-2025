import numpy as np
from matplotlib import pyplot as plt
from conjgrad import conjgrad
plt.ion()

def apply_bcs(V):
    #square coax
    n=V.shape[0]
    nn=n//2 #center at n/2
    w=n//8 #box width is n/8
    V[:,0]=0
    V[:,-1]=0
    V[0,:]=0
    V[-1,:]=0

    V[nn-w:nn+w+1,nn+w]=1
    V[nn-w:nn+w+1,nn-w]=1
    V[nn-w,nn-w:nn+w+1]=1
    V[nn+w,nn-w:nn+w+1]=1

def get_mask(n):
    mat=np.zeros([n,n])+np.nan
    apply_bcs(mat)
    return np.isfinite(mat) #this returns true everywhere the BCs were applied

def avg_neighbors(V):
    V=(np.roll(V,1,0)+np.roll(V,-1,0)+np.roll(V,1,1)+np.roll(V,-1,1))/4
    return V
def get_rho(V):
    return V-avg_neighbors(V)


def get_greens(n):
    x=np.fft.fftfreq(n)*n #gives us 0...n/2,-n/2..-1
    rsqr=np.outer(x**2,np.ones(n))
    rsqr=rsqr+rsqr.T #get dx and dy
    rsqr[0,0]=1 #dummy value to avoid annoying error message
    greens=np.log(rsqr)/2
    #fix the zero,zero value
    greens[0,0]=4*greens[1,0]-greens[1,1]-greens[1,-1]-greens[2,0]
    return greens/greens[0,0]

class LaplaceFT:
    def __init__(self,n):
        self.n=n
        self.bcs=np.zeros([n,n])
        apply_bcs(self.bcs)
        self.mask=get_mask(self.n)
        self.greens=get_greens(self.n*2)
        self.gft=np.fft.rfft2(self.greens)
        
    def get_rhs(self):
        return self.bcs
    def __matmul__(self,rho):
        tmp=np.zeros([2*self.n,2*self.n]) #embed in double-sized array
        tmp[:self.n,:self.n]=rho
        rhoft=np.fft.rfft2(tmp)
        Arho=np.fft.irfft2(rhoft*self.gft)
        return Arho[:self.n,:self.n]*self.mask
n=960
green=get_greens(n)
#plt.clf();plt.imshow(np.fft.fftshift(green));plt.show()

mat=LaplaceFT(n)
b=mat.get_rhs()

rho=conjgrad(mat,b,niter=60)
rho=rho*mat.mask

tmp=np.zeros([2*n,2*n]) #embed in double-sized array
tmp[:n,:n]=rho
rhoft=np.fft.rfft2(tmp)
Arho=np.fft.irfft2(rhoft*mat.gft)
V=Arho[:n,:n]
#V=np.fft.irfft2(np.fft.rfft2(rho)*mat.gft)
plt.clf()
plt.imshow(V)
plt.show()
