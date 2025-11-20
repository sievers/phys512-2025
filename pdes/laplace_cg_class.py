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


class Laplace:
    def __init__(self,n):
        self.n=n
        self.mask=get_mask(n)
        self.bcs=np.zeros([n,n])
        apply_bcs(self.bcs)
    def get_rhs(self):
        b=avg_neighbors(self.bcs) #smears out our boundary conditions
        b[self.mask]=0 #zero out the boundaries where we aren't actually solving
        return b
    def __matmul__(self,V):
        VV=V.copy()
        VV[self.mask]=0
        rho=VV-avg_neighbors(VV)
        rho[self.mask]=0
        return rho






n=120
mat=Laplace(n)
b=mat.get_rhs()
V=conjgrad(mat,b,niter=n*2)
V[mat.mask]=mat.bcs[mat.mask]
assert(1==0)    
def upres(V):
    n=V.shape[0]
    out=np.zeros([2*n,2*n])
    out[::2,::2]=V
    out[1::2,::2]=V
    out[::2,1::2]=V
    out[1::2,1::2]=V
    return out





nvec=[60,120,240,480,960]
for i in range(len(nvec)):
    n=nvec[i]
    if i==0:
        niter=20*n
    else:
        niter=200
    if i==0:
        V=np.zeros([n,n])
    else:
        V=upres(V) #V on the right is previous solution
    apply_bcs(V)
    plt.clf()
    plt.imshow(V)
    plt.show()
    #mask=get_mask(n)
        
    plt.figure(1)
    for i in range(niter):
        V=avg_neighbors(V)
        apply_bcs(V)
    plt.clf()
    plt.imshow(V)
    plt.pause(0.001)


plt.figure(2)
rho=get_rho(V)
plt.clf()
plt.imshow(rho)
plt.show()
rho_bad=rho.copy()
mask=get_mask(n)
rho_bad[mask]=0
print('total charge is ',np.sum(np.abs(rho)))
print('bad charge is ',np.sum(np.abs(rho_bad)))
